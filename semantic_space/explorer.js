/* Load pinned test evidence only when requested; keep the space graph fast. */
(() => {
  const loadButton = document.getElementById('load-explorer');
  const loadStatus = document.getElementById('explorer-status');
  loadButton.addEventListener('click', async () => {
    loadButton.disabled = true;
    loadStatus.textContent = '正在加载测试索引与证据…';
    try {
      const [catalogResponse, extraResponse] = await Promise.all([
        fetch('test_catalog.json'), fetch('explorer_extra.json')
      ]);
      if (!catalogResponse.ok || !extraResponse.ok) throw new Error('资料文件未能取得');
      const [catalog, extra] = await Promise.all([catalogResponse.json(), extraResponse.json()]);
      startExplorer(catalog, extra);
      loadButton.hidden = true;
      document.getElementById('test-search').disabled = false;
      loadStatus.textContent = '测试索引已加载。可搜索文件或测试函数。';
    } catch (error) {
      loadButton.disabled = false;
      loadStatus.textContent = `加载失败：${error.message}。可重试。`;
    }
  });

  function startExplorer(catalog, extra) {
  if (!catalog) return;
  const inheritedRows = extra.inherited;
  const inheritedPropertyRows = extra.inheritedProperty || [];
  const inheritedPropertyByTest = new Map(inheritedPropertyRows.map(row =>
    [`${row.snapshot}:${row.path}:${row.test_name}:${row.line}`, row]));
  const reviewedPropertyRows = extra.reviewedProperty || [];
  const reviewedPropertyByTest = new Map(reviewedPropertyRows.map(row =>
    [`${row.snapshot}:${row.path}:${row.test_name}:${row.line}`, row]));
  const helperRows = extra.helper;
  const helperByTest = new Map(helperRows.map(row =>
    [`${row.snapshot}:${row.test_path}:${row.test_name}:${row.test_line}`, row]));
  const callOracleRows = extra.callOracle;
  const collectionExceptions = extra.collectability;
  const collectionByTest = new Map(collectionExceptions.map(row =>
    [`${row.snapshot}:${row.path}:${row.test_name}:${row.line}`, row]));
  const dynamicRows = extra.dynamic;
  const nonstandardRows = extra.nonstandard;
  const nonstandardByTest = new Map(nonstandardRows.map(row =>
    [`${row.snapshot}:${row.path}:${row.test_name}:${row.line}`, row]));
  const sourceReviewRows = extra.sourceReview;
  const batchSourceReviewRows = extra.batchSourceReview;
  const batchSourceReviewByTest = new Map(batchSourceReviewRows.map(row =>
    [`${row.snapshot}:${row.path}:${row.test_name}:${row.line}`, row]));
  const batchSourceReviewByName = new Map(batchSourceReviewRows.map(row =>
    [`${row.snapshot}:${row.path}:${row.test_name}`, row]));
  const spaceAssessment = extra.spaceAssessment.assessments;
  const sourceReviewByTest = new Map(sourceReviewRows.map(row =>
    [`${row.snapshot}:${row.path}:${row.name}:${row.line}`, row]));
  const callOracleByTest = new Map();
  for (const row of callOracleRows) {
    const key = `${row.snapshot}:${row.path}:${row.test_name}:${row.test_line}`;
    if (!callOracleByTest.has(key)) callOracleByTest.set(key, []);
    callOracleByTest.get(key).push(row);
  }
  const inheritedByFile = new Map();
  for (const row of inheritedRows) {
    const key = `${row.snapshot}:${row.test_path}`;
    if (!inheritedByFile.has(key)) inheritedByFile.set(key, []);
    inheritedByFile.get(key).push(row);
  }
  const dynamicByFile = new Map();
  for (const row of dynamicRows) {
    const key = `${row.snapshot}:${row.path}`;
    if (!dynamicByFile.has(key)) dynamicByFile.set(key, []);
    dynamicByFile.get(key).push(row);
  }
  const space = JSON.parse(document.getElementById('space-data').textContent);
  const runtime = extra.runtime;
  const runtimeByTest = new Map();
  for (const entry of runtime.entries) {
    const key = `${entry.snapshot}:${entry.path}:${entry.name}`;
    if (!runtimeByTest.has(key)) runtimeByTest.set(key, []);
    runtimeByTest.get(key).push(entry);
  }
  const treeElement = document.getElementById('test-tree');
  const searchElement = document.getElementById('test-search');
  const searchResults = document.getElementById('test-search-results');
  const matchResult = document.getElementById('match-result');
  const banner = document.getElementById('selection-banner');
  const auxiliary = document.getElementById('aux-scopes');
  const records = [];
  const sourceNameCounts = new Map();
  const fileRecords = [];
  const trees = new Map();
  const areaIndex = new Map();

  function el(tag, className, label) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (label !== undefined) node.textContent = label;
    return node;
  }

  function createTreeNode() { return {dirs: new Map(), files: []}; }
  for (const snapshot of catalog.snapshots) {
    const root = createTreeNode();
    trees.set(snapshot.id, root);
    for (const file of snapshot.files) {
      for (const area of file.areas || []) areaIndex.set(area.id, area);
      let cursor = root;
      const parts = file.path.split('/');
      for (const segment of parts.slice(0, -1)) {
        if (!cursor.dirs.has(segment)) cursor.dirs.set(segment, createTreeNode());
        cursor = cursor.dirs.get(segment);
      }
      const entries = [];
      const inherited = inheritedByFile.get(`${snapshot.id}:${file.path}`) || [];
      const visibleItems = inherited.length
        ? file.items.filter(item => item.kind !== 'source_test_file_unresolved')
        : file.items;
      for (const item of visibleItems) {
        for (const area of item.classification?.areas || []) areaIndex.set(area.id, area);
        const id = records.length;
        const collectionKey = `${snapshot.id}:${file.path}:${item.name}:${item.line}`;
        const nonstandard = nonstandardByTest.get(collectionKey);
        const role = nonstandard?.status === 'script_main_invocation' ? 'script_test'
          : nonstandard?.status === 'explicit_host_invocation' ? 'helper_candidate'
          : nonstandard?.status === 'derived_test_class_found' ||
            nonstandard?.status === 'generated_test_class_found' ? 'class_base'
          : null;
        const collection = role
          ? {...nonstandard, source_role: role}
          : collectionByTest.get(collectionKey);
        const record = {id, snapshot: snapshot.id, path: file.path, item, collection};
        records.push(record);
        const key = `${record.snapshot}:${record.path}:${record.item.name}`;
        sourceNameCounts.set(key, (sourceNameCounts.get(key) || 0) + 1);
        entries.push(record);
      }
      for (const row of inherited) {
        const item = {name: `${row.test_class}::${row.method}`,
          line: Number(row.test_class_line), kind: 'python_inherited_test_candidate',
          definition_path: row.definition_path, definition_line: Number(row.definition_line),
          base_chain: row.base_chain, oracle_evidence: JSON.parse(row.oracle_evidence || '[]'),
          routes: JSON.parse(row.routes || '[]'),
          source_calls: JSON.parse(row.source_calls || '[]'),
          source_assertions: JSON.parse(row.source_assertions || '[]'),
          source_call_count: Number(row.source_call_count || 0),
          source_assertion_count: Number(row.source_assertion_count || 0),
          classification: JSON.parse(row.classification || '{}')};
        const record = {id: records.length, snapshot: snapshot.id, path: file.path,
          item, inherited: true};
        records.push(record);
        entries.push(record);
      }
      for (const row of dynamicByFile.get(`${snapshot.id}:${file.path}`) || []) {
        const source = file.items.find(item =>
          item.name === `${row.base_class}::${row.method}` &&
          Number(item.line) === Number(row.method_line));
        const item = {name: `${row.base_class}〔配置 ${Number(row.config_index) + 1}〕::${row.method}`,
          line: Number(row.method_line), kind: 'python_dynamic_test_candidate',
          classification: source?.classification || {}, routes: [],
          source_calls: source?.source_calls || [],
          source_assertions: source?.source_assertions || []};
        const record = {id: records.length, snapshot: snapshot.id, path: file.path,
          item, dynamic: row};
        records.push(record);
        entries.push(record);
      }
      const fileRecord = {snapshot: snapshot.id, path: file.path, file, entries};
      fileRecords.push(fileRecord);
      cursor.files.push({name: parts.at(-1), ...fileRecord});
    }
  }

  function testButton(record) {
    const role = record.collection?.source_role;
    const roleLabel = role === 'fixture' ? '〔fixture〕' :
      role === 'helper' || role === 'helper_candidate' ? '〔辅助〕' :
      role === 'class_base' ? '〔测试基类〕' :
      role === 'script_test' ? '〔脚本测试〕' :
      role === 'collection_review' ? '〔待核对收集〕' : '';
    const button = el('button', `test-entry${role && role !== 'collection_review' ? ' support' : ''}`,
      (record.dynamic ? '〔动态候选〕' : roleLabel) + record.item.name);
    button.type = 'button';
    button.title = record.inherited || record.dynamic
      ? `${record.path}:${record.item.line}:${record.item.name}`
      : `${record.path}:${record.item.line}`;
    button.addEventListener('click', () => selectTest(record));
    return button;
  }

  function directoryNode(name, child, initiallyOpen = false) {
    const details = el('details');
    details.append(el('summary', '', name));
    const populate = () => {
      if (details.dataset.rendered) return;
      details.dataset.rendered = 'true';
      for (const [subname, subchild] of [...child.dirs].sort((a, b) => a[0].localeCompare(b[0]))) {
        details.append(directoryNode(subname, subchild));
      }
      for (const file of child.files.sort((a, b) => a.name.localeCompare(b.name))) {
        details.append(fileNode(file));
      }
    };
    details.addEventListener('toggle', () => { if (details.open) populate(); });
    if (initiallyOpen) { details.open = true; populate(); }
    return details;
  }

  function fileNode(entry) {
    if (entry.entries.length === 0) {
      const label = el('button', 'file-entry support', entry.name);
      label.type = 'button';
      label.title = `${entry.file.path}（测试目录中的辅助文件或配置）`;
      label.addEventListener('click', () => selectFile(entry));
      return label;
    }
    const details = el('details');
    const summary = el('summary', '', `${entry.name}  (${entry.entries.length})`);
    summary.title = entry.file.path;
    summary.addEventListener('click', () => selectFile(entry));
    details.append(summary);
    details.addEventListener('toggle', () => {
      if (!details.open || details.dataset.rendered) return;
      details.dataset.rendered = 'true';
      for (const record of entry.entries) details.append(testButton(record));
    });
    return details;
  }

  for (const snapshot of catalog.snapshots) {
    const root = directoryNode(`${snapshot.title}  (${snapshot.files.length} 文件)`, trees.get(snapshot.id), true);
    treeElement.append(root);
  }
  const fileByKey = new Map(fileRecords.map(record => [`${record.snapshot}:${record.path}`, record]));

  function runtimeFor(record) {
    if (record.inherited || record.dynamic) return [];
    const key = `${record.snapshot}:${record.path}:${record.item.name}`;
    const entries = runtimeByTest.get(key) || [];
    if (sourceNameCounts.get(key) === 1) return entries;
    const first = record.item.start_line ?? record.item.line;
    return entries.filter(entry => Number.isInteger(entry.test_definition_line) &&
      first <= entry.test_definition_line && entry.test_definition_line <= record.item.line);
  }

  let searchTimer;
  searchElement.addEventListener('input', () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      const query = searchElement.value.trim().toLowerCase();
      searchResults.replaceChildren();
      treeElement.hidden = Boolean(query);
      searchResults.hidden = !query;
      if (!query) return;
      const files = fileRecords.filter(record => record.path.toLowerCase().includes(query));
      const tests = records.filter(record => record.item.name.toLowerCase().includes(query));
      searchResults.append(el('div', 'search-count', `${files.length.toLocaleString()} 个文件，${tests.length.toLocaleString()} 个名称匹配的测试；各显示前 100 个`));
      for (const record of files.slice(0, 100)) {
        const button = el('button', 'file-entry', `${record.snapshot} / ${record.path}`);
        button.type = 'button';
        button.title = record.path;
        button.addEventListener('click', () => selectFile(record));
        searchResults.append(button);
      }
      for (const record of tests.slice(0, 100)) {
        const button = testButton(record);
        button.textContent = `${record.snapshot} / ${record.path} :: ${button.textContent}`;
        searchResults.append(button);
      }
    }, 120);
  });

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, char =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
  }

  function markSelected(record, line) {
    document.querySelectorAll('[aria-selected=true]').forEach(node =>
      node.setAttribute('aria-selected', 'false'));
    const title = record.inherited || record.dynamic
      ? `${record.path}:${line}:${record.item.name}`
      : line ? `${record.path}:${line}` : record.path;
    document.querySelectorAll('.test-entry,.file-entry,summary').forEach(node => {
      if (node.title === title) node.setAttribute('aria-selected', 'true');
    });
  }

  function sourceLink(snapshot, path, line) {
    const snap = space.snapshots[snapshot];
    const repo = snapshot === 'vllm' ? 'vllm-project/vllm' : 'sgl-project/sglang';
    const url = `https://github.com/${repo}/blob/${snap.commit}/${path}#L${line}`;
    return `<a href="${url}" target="_blank" rel="noopener">${escapeHtml(snapshot + '/' + path + ':' + line)}</a>`;
  }

  function progressLink(record) {
    const query = record.path + (record.item ? ' ' + record.item.name : '');
    return `<p><a href="../review_progress/?q=${encodeURIComponent(query)}">在归类进度页查找此${record.item ? '测试' : '文件'}</a></p>`;
  }

  function helperEvidence(record) {
    const item = record.item;
    const row = helperByTest.get(`${record.snapshot}:${record.path}:${item.name}:${item.line}`);
    if (!row) return '';
    const sites = row.oracle_sites || [];
    const preview = sites.slice(0, 6).map(site =>
      `${sourceLink(record.snapshot, site.path, site.line)} ${escapeHtml(site.expression)}`).join('；');
    const status = sites.length
      ? `<p>辅助函数中的候选判定位置（展示 ${Math.min(sites.length, 6)}/${sites.length}）：${preview}。</p>`
      : '<p>尚未在可静态解析的测试辅助函数中找到判定位置。</p>';
    const limit = row.truncated ? '<p>辅助函数遍历达到深度或数量限制，仍有调用未审查。</p>' : '';
    const unresolved = row.unresolved_calls
      ? `<p>未解析的辅助调用：${escapeHtml(row.unresolved_calls)}。</p>` : '';
    return status + limit + unresolved +
      '<p>辅助函数位置是源码线索；是否执行、检查哪个产品行为，仍须结合输入与调用记录核对。</p>';
  }

  function showPlacement(entries, areas = [], selectionLabel = '', reviewedUnitIds = null) {
    document.getElementById('details').hidden = true;
    document.querySelector('.inspector').scrollTop = 0;
    document.querySelectorAll('#trace-filters button').forEach(button =>
      button.setAttribute('aria-pressed', 'false'));
    const active = new Set(entries.flatMap(entry => Object.keys(entry.observed_units)));
    const confirmed = new Set(entries.flatMap(entry => Object.keys(entry.confirmed_units || {})));
    const candidatePhases = new Set(areas.filter(area => area.phase).map(area => area.phase));
    const candidateAuxiliary = new Set(areas.filter(area => !area.phase).map(area => area.id));
    const sourceReviewedUnits = reviewedUnitIds === null ? null : new Set(reviewedUnitIds);
    document.querySelectorAll('.phase-heading').forEach(node =>
      node.classList.toggle('candidate', candidatePhases.has(node.dataset.phase)));
    document.querySelectorAll('.aux-scope').forEach(node =>
      node.classList.toggle('active', candidateAuxiliary.has(node.dataset.area)));
    auxiliary.classList.toggle('visible', candidateAuxiliary.size > 0);
    document.querySelectorAll('.unit').forEach(node => {
      node.classList.toggle('active', active.has(node.dataset.unit));
      node.classList.toggle('confirmed', confirmed.has(node.dataset.unit));
      const reviewed = sourceReviewedUnits?.has(node.dataset.unit) || false;
      node.classList.toggle('reviewed', reviewed && !confirmed.has(node.dataset.unit));
      const candidate = active.size === 0 && sourceReviewedUnits === null &&
        candidatePhases.has(node.dataset.phase);
      node.classList.toggle('candidate', candidate);
      node.classList.toggle('dimmed', (active.size > 0 || sourceReviewedUnits !== null || candidatePhases.size > 0) &&
        !active.has(node.dataset.unit) && !reviewed && !candidate);
    });
    document.querySelectorAll('.edge').forEach(node => {
      node.classList.remove('active');
      node.classList.toggle('dimmed', active.size > 0 &&
        (!active.has(node.dataset.parent) || !active.has(node.dataset.child)));
    });
    const prefix = selectionLabel ? `${selectionLabel}｜` : '';
    banner.textContent = prefix + (entries.length === 0
      ? sourceReviewedUnits !== null
        ? `已核对测试断言；相关验证单元 ${[...sourceReviewedUnits].join('、') || '尚未建立'}。运行路径待追踪。`
        : `验证对象候选：${areas.map(area => area.label).join('、') || '待核查'}。尚无运行追踪，具体路径与检查性质待确认。`
      : active.size === 0
        ? `验证对象候选：${areas.map(area => area.label).join('、') || '待核查'}。已运行追踪，但尚未命中图中源码锚点。`
        : `验证对象候选：${areas.map(area => area.label).join('、') || '待核查'}；实际执行命中 ${active.size} 个源码锚点，其中 ${confirmed.size} 个已关联到具体判定。`);
    banner.classList.add('visible');
    matchResult.classList.add('visible');
  }

  function selectTest(record) {
    if (record.inherited) { selectInheritedTest(record); return; }
    if (record.dynamic) {
      markSelected(record, record.item.line);
      const fileRecord = fileByKey.get(`${record.snapshot}:${record.path}`);
      const review = batchSourceReviewByName.get(
        `${record.snapshot}:${record.path}:${record.dynamic.base_class}::${record.dynamic.method}`);
      const reviewedUnits = review ? review.unit_ids.split(';').filter(Boolean) : null;
      showPlacement([], record.item.classification?.areas || fileRecord?.file.areas || [],
        record.item.name, reviewedUnits);
      banner.textContent = `动态生成候选：第 ${Number(record.dynamic.config_index) + 1} 组配置使用 ${record.dynamic.base_class}.${record.dynamic.method}；` +
        (review ? '方法体断言已审阅，具体配置的收集、跳过和执行待核对。' : '方法体性质及具体配置的执行待核对。');
      const status = review?.status === 'mapped' ? '可放入当前验证单元' :
        review?.status === 'needs_refinement' ? '当前空间需要细化' : '待审阅';
      const reviewedEvidence = review
        ? `<p>方法体检查：${escapeHtml(review.checked_property)}</p>` +
          `<p>空间对照：${status}；相关单元：${escapeHtml(review.unit_ids || '尚未建立')}。</p>` +
          (review.oracle_lines
            ? `<p>直接断言：${review.oracle_lines.split(';').filter(Boolean).map(line =>
                sourceLink(record.snapshot, record.path, Number(line))).join('、')}。</p>` : '') +
          (JSON.parse(review.supporting_sources || '[]').length
            ? `<p>辅助判定实现：${JSON.parse(review.supporting_sources).map(site =>
                sourceLink(record.snapshot, site.path, Number(site.line))).join('、')}。</p>` : '') +
          `<p>适用条件：${escapeHtml(review.applicability_note)}</p>` : '';
      matchResult.innerHTML = `<h2>${escapeHtml(record.item.name)}</h2>` +
        `<p>方法源码：${sourceLink(record.snapshot, record.path, record.item.line)}</p>` +
        `<p>配置源码：${sourceLink(record.snapshot, record.path, Number(record.dynamic.config_line))}；` +
        `生成类的循环：${sourceLink(record.snapshot, record.path, Number(record.dynamic.loop_line))}</p>` +
        reviewedEvidence +
        `<p>候选生成表达式：${escapeHtml(record.dynamic.class_name_expression)}。` +
        `生成后的类名、参数实例和运行判定另行核对。</p>` + progressLink(record);
      return;
    }
    if (record.collection && !['collection_review', 'script_test'].includes(record.collection.source_role)) {
      markSelected(record, record.item.line);
      showPlacement([], [], record.item.name);
      banner.textContent = `源码定义：${record.item.name}；${record.collection.status}。该定义当前不作为独立测试入口映射到验证单元。`;
      const review = record.collection.review_evidence?.split(':');
      const reviewLink = review?.length === 2
        ? `<p>用途证据：${sourceLink(record.snapshot, review[0], Number(review[1]))}</p>` : '';
      matchResult.innerHTML = `<h2>${escapeHtml(record.item.name)}</h2><p>源码位置：${sourceLink(record.snapshot, record.path, record.item.line)}</p><p>收集身份：${escapeHtml(record.collection.status)}；该定义作为其他测试的基类或辅助调用使用。</p>` + reviewLink + progressLink(record);
      return;
    }
    const fileRecord = fileByKey.get(`${record.snapshot}:${record.path}`);
    const classification = record.item.classification || {};
    const candidateAreas = classification.areas || fileRecord?.file.areas || [];
    const reviewed = sourceReviewByTest.get(
      `${record.snapshot}:${record.path}:${record.item.name}:${record.item.line}`);
    const batchReviewed = batchSourceReviewByTest.get(
      `${record.snapshot}:${record.path}:${record.item.name}:${record.item.line}`);
    const sourceChecked = reviewedPropertyByTest.get(
      `${record.snapshot}:${record.path}:${record.item.name}:${record.item.line}`);
    const assessed = reviewed && spaceAssessment[String(reviewed.sample_index)];
    const areas = reviewed
      ? reviewed.reviewed_areas.map(area => areaIndex.get(area)).filter(Boolean)
      : candidateAreas;
    const entries = runtimeFor(record);
    markSelected(record, record.item.line);
    showPlacement(entries, areas, record.item.name,
      sourceChecked ? sourceChecked.unit_id.split(';').filter(Boolean)
      : assessed ? assessed.unit_ids : batchReviewed ? batchReviewed.unit_ids.split(';').filter(Boolean) : null);
    if (batchReviewed && entries.length === 0) {
      banner.textContent = `${record.item.name}｜方法体断言已审阅；${batchReviewed.status === 'mapped'
        ? '相关验证单元已定位' : batchReviewed.status === 'needs_refinement'
          ? '当前验证单元仍需细化' : '不属于当前请求验证空间'}。实际执行路径待核对。`;
    }
    const evidence = record.item.routes.flatMap(match => match.evidence);
    const observed = [...new Set(entries.flatMap(entry => Object.keys(entry.observed_units)))];
    const confirmed = [...new Set(entries.flatMap(entry => Object.keys(entry.confirmed_units || {})))];
    const unlinked = [...new Set(entries.flatMap(entry => entry.unlinked_observed_units || []))];
    const reasonLabels = {
      test_did_not_pass: '运行未通过',
      trace_truncated: '运行追踪截断',
      trace_instrumentation_error: '追踪采集出错',
      no_space_anchor_hit: '调用了产品代码，但未命中当前空间的源码锚点',
      no_observed_python_product_call: '未观察到当前进程中的 Python 产品代码调用',
      source_hit_without_proven_oracle_relation: '命中源码，但部分判定尚未与该调用建立关系',
    };
    const reasons = [...new Set(entries.flatMap(entry => entry.unconfirmed_reasons || []))];
    const confirmations = entries.flatMap(entry => Object.entries(entry.confirmed_units || {}).map(([unit, links]) =>
      `${unit}: ${links.map(link => `${link.field}=${link.expected}`).join('，')}`));
    const hits = entries.flatMap(entry => Object.values(entry.observed_units).flat());
    const chainObservation = hits.flatMap(hit => hit.observations || []).find(observation => observation.call_chain?.length);
    const chain = chainObservation?.call_chain || [];
    const anchors = [...new Set(entries.flatMap(entry => Object.keys(entry.oracle_anchor_hits)))];
    const sites = entries.flatMap(entry => entry.executed_assertions || []);
    const calls = entries.flatMap(entry => entry.product_calls || []);
    const sourceCalls = record.item.source_calls || [];
    const sourceAssertions = record.item.source_assertions || [];
    const callOracleLinks = callOracleByTest.get(
      `${record.snapshot}:${record.path}:${record.item.name}:${record.item.line}`) || [];
    const basisLabels = {
      repository_infrastructure: '仓库 CI 或工具目录',
      protocol_test_module: '协议测试模块',
      protocol_test_name: '协议测试名称',
      component_test_directory: '组件测试目录',
      file_only_path: '文件级路径',
      task_quality_marker: '任务评测名称',
      named_direct_product_call: '测试名称与直接调用',
      direct_product_call: '源码中的直接调用',
      test_name: '测试名称',
      specific_path: '测试路径',
      file_path: '文件路径',
      path_fallback: '文件路径',
      unresolved: '待源码复核',
    };
    const callClues = sourceCalls.slice(0, 8).map(call =>
      `${sourceLink(record.snapshot, record.path, call.line)} ${escapeHtml(call.target)}`).join('；');
    const assertionClues = sourceAssertions.map(site =>
      `${sourceLink(record.snapshot, record.path, site.line)} ${escapeHtml(site.expression)}`).join('；');
    const linkedClues = callOracleLinks.slice(0, 4).map(link =>
      `${sourceLink(record.snapshot, record.path, link.product_call_line)} ${escapeHtml(link.product_target)} → ` +
      `${sourceLink(record.snapshot, record.path, link.oracle_line)} ${escapeHtml(link.oracle_expression)}` +
      (link.via_variable ? `（经 ${escapeHtml(link.via_variable)}）` : '')).join('；');
    const reviewedEvidence = reviewed
      ? `<p>已回读源码与断言：${escapeHtml(reviewed.checked_property)}。判定位置：` +
        [...(reviewed.oracle_lines || []).map(line => ({path: record.path, line})),
          ...(reviewed.oracle_sites || [])]
          .map(site => sourceLink(record.snapshot, site.path, site.line)).join('、') +
        '。这是源码核对结论，执行条件仍须由运行记录确认。</p>'
      : '';
    const batchReviewedEvidence = batchReviewed
      ? `<p>已回读源码与断言：${escapeHtml(batchReviewed.checked_property)}</p>` +
        (batchReviewed.oracle_lines
          ? `<p>直接判定位置：${batchReviewed.oracle_lines.split(';').filter(Boolean).map(line =>
              sourceLink(record.snapshot, record.path, Number(line))).join('、')}。</p>` : '') +
        (JSON.parse(batchReviewed.supporting_sources || '[]').length
          ? `<p>辅助判定实现：${JSON.parse(batchReviewed.supporting_sources).map(site =>
              sourceLink(record.snapshot, site.path, Number(site.line))).join('、')}。</p>` : '') +
        `<p>空间对照：${escapeHtml({mapped: '可放入当前验证单元', needs_refinement: '当前空间需细化', outside_request_verification: '位于当前请求验证空间之外'}[batchReviewed.status] || batchReviewed.status)}；相关单元：${escapeHtml(batchReviewed.unit_ids || '尚未建立')}。` +
        `适用条件：${escapeHtml(batchReviewed.applicability_note)}</p>` : '';
    const assessmentLabels = {
      mapped: '当前单元可容纳这项局部性质',
      needs_refinement: '当前空间需细化这项性质',
      outside_request_verification: '该性质属于请求验证空间以外的测试',
    };
    const spaceEvidence = assessed
      ? `<p>空间对照：${escapeHtml(assessmentLabels[assessed.status] || assessed.status)}` +
        (assessed.unit_ids.length ? `；相关单元 ${escapeHtml(assessed.unit_ids.join('、'))}` : '') +
        (assessed.remaining ? `；${escapeHtml(assessed.remaining)}` : '') +
        '。这是单元边界审阅，不代表运行覆盖或完整单元证明。</p>'
      : '';
    const sourceCheckedEvidence = sourceChecked
      ? `<p>已核对的局部检查：${escapeHtml(sourceChecked.property)}</p>` +
        `<p>候选单元：${escapeHtml(sourceChecked.unit_id)}；适用条件：${escapeHtml(sourceChecked.conditions)}</p>` +
        `<p>断言源码：${sourceChecked.oracle_source.split(';').map(location => {
          const cut = location.lastIndexOf(':');
          return sourceLink(record.snapshot, location.slice(0, cut), Number(location.slice(cut + 1)));
        }).join('、')}。这不代表已运行或整个单元已证明。</p>`
      : '';
    matchResult.innerHTML = `<h2>${escapeHtml(record.item.name)}</h2>` +
      `<p>${sourceLink(record.snapshot, record.path, record.item.line)}</p>` +
      (record.item.kind === 'source_test_file_unresolved' ? '<p>该条是文件级待解析记录，尚未识别为可执行的测试函数。</p>' : '') +
      sourceCheckedEvidence +
      reviewedEvidence +
      batchReviewedEvidence +
      spaceEvidence +
      `<p>程序归类候选：${escapeHtml(candidateAreas.map(area => area.label).join('、') || '待归类')}；依据：${escapeHtml(basisLabels[classification.basis] || classification.basis || '文件路径')}${classification.ambiguous ? '；多个候选得分相同，待核查' : ''}。</p>` +
      `<p>运行实例 ${entries.length} 项；执行到的断言位置 ${sites.length} 处。</p>` +
      (sourceCalls.length ? `<p>源码中的直接产品调用（静态线索，展示前 ${Math.min(sourceCalls.length, 8)}/${record.item.source_call_count}）：${callClues}</p>` : '') +
      (sourceAssertions.length ? `<p>源码中的断言或预期异常（静态线索，${sourceAssertions.length}/${record.item.source_assertion_count}）：${assertionClues}</p>` : '') +
      (callOracleLinks.length ? `<p>产品调用到判定表达式的源码关联（展示 ${Math.min(callOracleLinks.length, 4)}/${callOracleLinks.length}）：${linkedClues}。</p>` : '') +
      (!sourceAssertions.length ? helperEvidence(record) : '') +
      (entries.some(entry => entry.execution_mode?.startsWith('isolated_')) ? '<p>这批记录来自隔离 CPU 运行：官方测试体和目标产品模块保持原样，框架初始化及无关测试辅助模块由轻量入口替代。</p>' : '') +
      (calls.length ? `<p>实际调用：${escapeHtml(calls.slice(0, 3).map(call => `${call.path}:${call.line} ${call.function}`).join('；'))}${calls.length > 3 ? '……' : ''}</p>` : '') +
      (observed.length ? `<p>实际经过的验证单元源码：${escapeHtml(observed.join('、'))}。</p>` : '') +
      (hits.length ? `<p>命中位置与条件：${escapeHtml(hits.slice(0, 3).map(hit => `${hit.path}:${hit.line} ${JSON.stringify(hit.guard)}`).join('；'))}。</p>` : '') +
      (chain.length ? `<p>该源码命中时的 Python 调用链：${escapeHtml(chain.map(frame => `${frame.function} (${frame.path}:${frame.line})`).join(' → '))}${chainObservation.call_chain_truncated ? ' …' : ''}。</p>` : '') +
      (confirmed.length ? `<p>检查结果已关联：${escapeHtml(confirmations.join('；'))}。</p>` : '') +
      (unlinked.length ? `<p>执行已命中、判定尚未关联：${escapeHtml(unlinked.join('、'))}。</p>` : '') +
      (reasons.length ? `<p>待核查：${escapeHtml(reasons.map(reason => reasonLabels[reason] || reason).join('；'))}。</p>` : '') +
      (anchors.length ? `<p>执行到已登记的判定锚点：${escapeHtml(anchors.join('、'))}。</p>` : '') +
      (sites.length ? `<p>断言例：${escapeHtml(sites.slice(0, 3).map(site => `${site.path}:${site.line} ${site.expression}`).join('；'))}</p>` : '') +
      (evidence.length ? `<p>静态名称线索：${escapeHtml([...new Set(evidence)].join('、'))}。</p>` : '') +
      '<p>未标为“检查结果已关联”的性质仍待核查；原生代码及子进程不在这份 Python 追踪中。</p>' + progressLink(record);
  }

  function selectInheritedTest(record) {
    const item = record.item;
    const fileRecord = fileByKey.get(`${record.snapshot}:${record.path}`);
    const areas = item.classification.areas || fileRecord?.file.areas || [];
    const review = batchSourceReviewByTest.get(
      `${record.snapshot}:${record.path}:${item.name}:${item.line}`);
    const checked = inheritedPropertyByTest.get(
      `${record.snapshot}:${record.path}:${item.name}:${item.line}`);
    const reviewedUnits = review?.status === 'skipped_by_class' ? []
      : checked ? checked.unit_id.split(';').filter(Boolean)
      : review ? review.unit_ids.split(';').filter(Boolean) : null;
    markSelected(record, item.line);
    showPlacement([], areas, item.name, reviewedUnits);
    if (review) banner.textContent = review.status === 'skipped_by_class'
      ? `${item.name}｜测试类被显式跳过；此方法未执行验证。`
      : `${item.name}｜继承方法及具体类的源码已核对；收集、执行与运行判定仍待确认。`;
    const assertions = item.oracle_evidence.slice(0, 8).map(site =>
      `${sourceLink(record.snapshot, item.definition_path, site.line)} ${escapeHtml(site.expression)}`).join('；');
    const checkedEvidence = checked
      ? `<p>公共方法中已核对的局部检查：${escapeHtml(checked.property)}</p>` +
        `<p>候选单元：${escapeHtml(checked.unit_id)}；适用条件：${escapeHtml(checked.conditions)}</p>` +
        `<p>断言或辅助判定源码：${checked.oracle_source.split(';').map(location => {
          const cut = location.lastIndexOf(':');
          return sourceLink(record.snapshot, location.slice(0, cut), Number(location.slice(cut + 1)));
        }).join('、')}。</p>` +
        (review?.status === 'skipped_by_class' ? '<p>具体测试类被显式跳过；这条源码检查未执行。</p>' : '')
      : '';
    const reviewedEvidence = review && !checked
      ? `<p>已核对的检查：${escapeHtml(review.checked_property)}</p>` +
        `<p>空间对照：${escapeHtml({mapped: '可放入当前验证单元', needs_refinement: '当前空间需要细化', outside_request_verification: '位于当前请求空间之外', skipped_by_class: '测试类显式跳过'}[review.status] || review.status)}；相关单元：${escapeHtml(review.unit_ids || '尚未建立')}。</p>` +
        `<p>条件与边界：${escapeHtml(review.applicability_note)}</p>` +
        (JSON.parse(review.supporting_sources || '[]').length
          ? `<p>辅助判定：${JSON.parse(review.supporting_sources).map(site =>
              sourceLink(record.snapshot, site.path, Number(site.line))).join('、')}。</p>` : '')
      : '';
    matchResult.innerHTML = `<h2>${escapeHtml(item.name)}</h2>` +
      `<p>具体测试类：${sourceLink(record.snapshot, record.path, item.line)}；继承方法：${sourceLink(record.snapshot, item.definition_path, item.definition_line)}。</p>` +
      `<p>静态继承链：${escapeHtml(item.base_chain)}。</p>` +
      checkedEvidence + reviewedEvidence +
      (assertions ? `<p>方法体的断言源码：${assertions}</p>` : helperEvidence(record)) +
      (checked ? '<p>这是源码级局部断言记录；具体 pytest 收集、执行路径和运行判定另行核对。</p>'
        : '<p>这是继承方法的源码候选；具体 pytest 收集、执行路径和判定关系尚待核对。</p>') + progressLink(record);
  }

  function selectFile(record) {
    markSelected(record);
    const entries = record.entries.flatMap(item => runtimeFor(item));
    const areas = record.file.areas || [];
    showPlacement(entries, areas, record.path.split('/').at(-1));
    const status = record.entries.length
      ? `识别到 ${record.entries.length} 个源码级测试入口；点击文件展开后可逐项查看。`
      : '该文件位于测试相关目录，当前未识别到独立测试函数，按辅助文件或测试配置处理。';
    const areaBasis = areas.length
      ? `<p>文件主题线索：${escapeHtml(areas.map(area => area.label).join('、'))}。</p>`
      : '';
    matchResult.innerHTML = `<h2>${escapeHtml(record.path.split('/').at(-1))}</h2>` +
      `<p>${sourceLink(record.snapshot, record.path, 1)}</p><p>${status} 已追踪 ${entries.length} 个运行实例。</p>` +
      areaBasis +
      '<p>图中只标亮追踪到的源码锚点；断言性质仍需逐项关联。</p>' + progressLink(record);
  }
  }
})();
