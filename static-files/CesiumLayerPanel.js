// CesiumLayerPanel.js  — ESM module
// Minimal, robust layer panel for Cesium DataSources with single click delegation.

export default class CesiumLayerPanel {
  /**
   * @param {Cesium.Viewer} viewer
   * @param {Object} [opts]
   * @param {string|HTMLElement} [opts.panel]         Selector or element for the panel container (created if missing).
   * @param {string} [opts.listSelector]              Selector for the inner list (created if missing). Default '#dsList'.
   * @param {(src: any)=>boolean} [opts.filter]       Only show DataSources where filter(src) === true.
   * @param {string} [opts.title]                     Panel title. Default 'Data Sources'.
   */
  constructor(viewer, opts = {}) {
    if (!viewer) throw new Error('CesiumLayerPanel: viewer is required');
    this.viewer = viewer;

    // Options
    this.opts = {
      panel: '#controlPanel',
      listSelector: '#dsList',
      filter: null,              // e.g., (src) => src.__wedge === true
      title: 'Data Sources',
      ...opts,
    };

    // DOM
    this.panel = this._resolvePanel(this.opts.panel);
    this.list = this._resolveList(this.panel, this.opts.listSelector);

    // State
    this._idToSource = new Map();
    this._perSourceListeners = new Map(); // DataSource -> { onLoading, onChanged }
    this._uid = 0;
    this._renderScheduled = false;

    // Bind handlers so we can remove them later
    this._onPanelClick = this._onPanelClick.bind(this);
    this._onDataSourceAdded = this._onDataSourceAdded.bind(this);
    this._onDataSourceRemoved = this._onDataSourceRemoved.bind(this);

    // Wire events
    this.panel.addEventListener('click', this._onPanelClick);
    const coll = this.viewer.dataSources;
    coll.dataSourceAdded.addEventListener(this._onDataSourceAdded);
    coll.dataSourceRemoved.addEventListener(this._onDataSourceRemoved);

    // Attach listeners to existing sources (if any)
    for (let i = 0; i < coll.length; i++) {
      const src = coll.get(i);
      this._attachSourceListeners(src);
    }

    // Initial paint
    this._render();
  }

  /** Force a re-render of the panel. */
  refresh() { this._render(); }

  /** Clean up all listeners and DOM references created by this panel. */
  destroy() {
    // Remove collection listeners
    const coll = this.viewer.dataSources;
    coll.dataSourceAdded.removeEventListener(this._onDataSourceAdded);
    coll.dataSourceRemoved.removeEventListener(this._onDataSourceRemoved);

    // Remove per-source listeners
    for (const [src, fns] of this._perSourceListeners.entries()) {
      if (fns.onLoading && src?.loadingEvent?.removeEventListener) {
        src.loadingEvent.removeEventListener(fns.onLoading);
      }
      if (fns.onChanged && src?.changedEvent?.removeEventListener) {
        src.changedEvent.removeEventListener(fns.onChanged);
      }
    }
    this._perSourceListeners.clear();

    // Remove panel listener (keep the element in DOM)
    this.panel.removeEventListener('click', this._onPanelClick);

    // Clear UI state maps
    this._idToSource.clear();
  }

  // -------------------- private helpers --------------------

  _resolvePanel(panelOpt) {
    let el = null;
    if (typeof panelOpt === 'string') el = document.querySelector(panelOpt);
    else if (panelOpt instanceof HTMLElement) el = panelOpt;

    if (!el) {
      el = document.createElement('div');
      el.id = 'controlPanel';
      Object.assign(el.style, {
        position: 'absolute',
        top: '10px',
        left: '10px',
        background: 'rgba(255,255,255,0.9)',
        padding: '10px',
        borderRadius: '6px',
        boxShadow: '0 2px 6px rgba(0,0,0,0.2)',
        fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, sans-serif',
        fontSize: '14px',
        zIndex: 1000,
      });
      const title = document.createElement('strong');
      title.textContent = this.opts.title || 'Data Sources';
      title.style.display = 'block';
      title.style.marginBottom = '8px';
      el.appendChild(title);
      document.body.appendChild(el);
    } else {
      // Update title if present, else add one
      const first = el.querySelector('strong');
      if (first) first.textContent = this.opts.title || 'Data Sources';
      else {
        const title = document.createElement('strong');
        title.textContent = this.opts.title || 'Data Sources';
        title.style.display = 'block';
        title.style.marginBottom = '8px';
        el.prepend(title);
      }
    }
    return el;
  }

  _resolveList(panel, listSelector) {
    const sel = listSelector || '#dsList';
    let list = panel.querySelector(sel);
    if (!list) {
      list = document.createElement('div');
      list.id = sel.startsWith('#') ? sel.slice(1) : sel;
      panel.appendChild(list);
    }
    return list;
  }

  _scheduleRender() {
    if (this._renderScheduled) return;
    this._renderScheduled = true;
    Promise.resolve().then(() => {
      this._renderScheduled = false;
      this._render();
    });
  }

  _render() {
    this._idToSource.clear();
    this.list.innerHTML = '';

    // Copy DataSourceCollection -> array
    const dsArray = [];
    const coll = this.viewer.dataSources;
    for (let i = 0; i < coll.length; i++) dsArray.push(coll.get(i));

    // Optional filter
    const filtered = typeof this.opts.filter === 'function'
      ? dsArray.filter(this.opts.filter)
      : dsArray;

    if (!filtered.length) {
      const empty = document.createElement('div');
      empty.textContent = 'No data sources loaded.';
      empty.style.opacity = '0.7';
      this.list.appendChild(empty);
      return;
    }

    const frag = document.createDocumentFragment();
    filtered.forEach((src, idx) => {
      const row = document.createElement('label');
      Object.assign(row.style, {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        margin: '6px 0',
      });

      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = src.show !== false;
      const id = `ds-${++this._uid}`;
      cb.dataset.dsuid = id;
      cb.setAttribute('aria-label', 'toggle data source visibility');

      const name = document.createElement('span');
      name.textContent = (src?.name && String(src.name).trim())
        ? String(src.name).trim()
        : (src?.constructor?.name || 'DataSource') + ' ' + (idx + 1);

      row.style.opacity = src?.isLoading ? '0.5' : '1';
      row.appendChild(cb);
      row.appendChild(name);
      frag.appendChild(row);

      this._idToSource.set(id, src);
    });

    this.list.appendChild(frag);
  }

  _attachSourceListeners(src) {
    const fns = {};
    if (src?.loadingEvent?.addEventListener) {
      fns.onLoading = () => this._scheduleRender();
      src.loadingEvent.addEventListener(fns.onLoading);
    }
    if (src?.changedEvent?.addEventListener) {
      fns.onChanged = () => this._scheduleRender();
      src.changedEvent.addEventListener(fns.onChanged);
    }
    if (Object.keys(fns).length) {
      this._perSourceListeners.set(src, fns);
    }
  }

  _detachSourceListeners(src) {
    const fns = this._perSourceListeners.get(src);
    if (!fns) return;
    if (fns.onLoading && src?.loadingEvent?.removeEventListener) {
      src.loadingEvent.removeEventListener(fns.onLoading);
    }
    if (fns.onChanged && src?.changedEvent?.removeEventListener) {
      src.changedEvent.removeEventListener(fns.onChanged);
    }
    this._perSourceListeners.delete(src);
  }

  _onDataSourceAdded(collection, src) {
    this._attachSourceListeners(src);
    this._scheduleRender();
  }

  _onDataSourceRemoved(collection, src) {
    this._detachSourceListeners(src);
    this._scheduleRender();
  }

  _onPanelClick(evt) {
    const t = evt.target;
    if (!(t instanceof HTMLInputElement) || t.type !== 'checkbox') return;
    const id = t.dataset.dsuid;
    if (!id) return;
    const src = this._idToSource.get(id);
    if (!src) return;
    src.show = !!t.checked;
    this.viewer.scene.requestRender();
  }
}
