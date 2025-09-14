    function getParam(name) {
      const source = window.location.search.length > 1
        ? window.location.search.substring(1)
        : window.location.hash.slice(1);
      const p = new URLSearchParams(source);
      return p.get(name);
    }  

    const mapurl2 = getParam("mapurl");
    console.log("CZML URL:", mapurl2);

    const elevfeet = getParam("elevfeet");
    console.log("Elevation path in feet is ", elevfeet);


// allow reassignment from the panel
  let maxFeet = 600;
  console.log("Elevation path in feet is ", elevfeet);
  if(elevfeet != null){
    maxFeet = elevfeet;
  }







// —— Chart cache: callsign+feet -> {canvas, callsign, feet} —— //
  const profileIndex = new Map();
  const keyFor = (callsign, feet) => `${sanitizeId(callsign)}::${feet}`;
  function sanitizeId(s) { return String(s || 'Unknown').replace(/[^A-Za-z0-9_-]/g, '_'); }

  function flash(el) {
    el.classList.add('profileFlash');
    setTimeout(() => el.classList.remove('profileFlash'), 650);
  }
  
// Find an entity by id across all loaded CZML data sources (and the default collection)
function findEntityById(id) {
  let e = viewer.entities.getById(id);
  if (e) return e;
  const dsCol = viewer.dataSources;
  for (let i = 0; i < dsCol.length; i++) {
    const ds = dsCol.get(i);
    e = ds.entities.getById(id);
    if (e) return e;
  }
  return null;
}


function getEntityInterval(entity) {
  // 1) entity-level availability (if present)
  const av = entity.availability;
  if (av && av.length > 0) {
    const interval = av.get(0);
    if (interval) return { start: interval.start, stop: interval.stop };
  }

  // 2) polyline.show TimeIntervalCollectionProperty
  const showProp = entity.polyline && entity.polyline.show;
  // Works when CZML used "show": [{ interval: "...", boolean: true }]
  if (showProp && showProp.intervals && showProp.intervals.length > 0) {
    const intv = showProp.intervals.get(0); // first interval
    if (intv) return { start: intv.start, stop: intv.stop };
  }

  // No time info
  return null;
}

////////////////////////////////////////////////////////////////
// Move the timeline/clock so the polyline is visible; fly the camera there
function focusEntityTime(entity) {
  if (!entity) return;
  viewer.clock.shouldAnimate = false;

  const interval = getEntityInterval(entity);
  if (interval) {
    const { start, stop } = interval;

    // Zoom the on-screen timeline if present
    if (viewer.timeline) viewer.timeline.zoomTo(start, stop);

    // Choose the midpoint time
    const mid = Cesium.JulianDate.addSeconds(
      start,
      Cesium.JulianDate.secondsDifference(stop, start) / 2,
      new Cesium.JulianDate()
    );
    viewer.clock.currentTime = mid;
  }

  // Ensure it’s on screen
  viewer.flyTo(entity, { duration: 0.8 }).catch(() => {});
}


////////////main block to initially set up viewer
////////////////////////////////////////////////////////////
const viewer = new Cesium.Viewer("cesiumContainer", { terrain: Cesium.Terrain.fromWorldTerrain(), });	
viewer.scene.globe.enableLighting = true;	
let czmlurl = 'https://raw.githubusercontent.com/hcarter333/rm-rbn-history/refs/heads/main/maps/OrganSteep.czml';
console.log(czmlurl);
if(mapurl2 != null){
  czmlurl = mapurl2;
}
console.log(czmlurl);
viewer.dataSources.add(Cesium.CzmlDataSource.load(czmlurl));


  // —— control panel logic —— //
  document.getElementById('loadBtn').addEventListener('click', () => {
    let mf = parseFloat(document.getElementById('maxFeetInput').value);
    if (isNaN(mf) || mf == 0) {
      mf = elevfeet;
      //maxFeet = mf;
    }
    let czmlUrl = document.getElementById('czmlInput').value;
    if(czmlUrl == ""){
      czmlInput = mapurl2
    }
    if (czmlUrl) {
      Cesium.CzmlDataSource.load(czmlUrl)
        .then(ds => viewer.dataSources.add(ds))
        .catch(err => console.error('Failed to load CZML:', err));
    }
  });
  document.getElementById('updateMaxBtn').addEventListener('click', () => {
    const mf = parseFloat(document.getElementById('maxFeetInput').value);
    if (!isNaN(mf) && mf > 0) {
      maxFeet = mf;
      console.log(`maxFeet is now ${maxFeet}`);
    } else {
      alert('Please enter a positive number for Max Feet.');
    }
  });

  viewer.selectedEntityChanged.addEventListener(async function(entity) {
    if (!entity || !entity.polyline) return;

    // 1) Get callsign from the CZML "id" (or .name)
    const callsign = entity.id || entity.name || 'Unknown';


  // If we already have a chart for (callsign, current maxFeet), scroll to it and bail.
  const currentFeet = maxFeet;
  const existing = profileIndex.get(keyFor(callsign, currentFeet));
  if (existing && existing.canvas && document.body.contains(existing.canvas)) {
    existing.canvas.scrollIntoView({ behavior: 'smooth', block: 'center' });
    flash(existing.canvas);
    return; // ← skip re-generation
  }

  // 2) Extract the two cartographic endpoints
    const positions = entity.polyline.positions.getValue(Cesium.JulianDate.now());
    if (!positions || positions.length < 2) return;
    const startCarto = Cesium.Cartographic.fromCartesian(positions[0]);
    const endCarto   = Cesium.Cartographic.fromCartesian(positions[1]);
    const startLat   = Cesium.Math.toDegrees(startCarto.latitude);
    const startLon   = Cesium.Math.toDegrees(startCarto.longitude);
    const endLat     = Cesium.Math.toDegrees(endCarto.latitude);
    const endLon     = Cesium.Math.toDegrees(endCarto.longitude);

    // 3) Sample the first maxFeet ft in 100 steps
    const geodesic = new Cesium.EllipsoidGeodesic(startCarto, endCarto);
    const totalDist = geodesic.surfaceDistance;
    const maxMeters = maxFeet * 0.3048;
    const count     = 100;
    const fetches   = [];
    const samples = [];   // ← NEW
    let last_lon = 0, last_lat = 0;

    for (let i = 0; i < count; i++) {
      const frac = (maxMeters / (count - 1) * i) / totalDist;
      const pos  = geodesic.interpolateUsingFraction(frac);
      const lon  = Cesium.Math.toDegrees(pos.longitude);
      const lat  = Cesium.Math.toDegrees(pos.latitude);
      last_lon = lon;
      last_lat = lat;
      samples.push({ lat, lon });   // ← NEW
      const url  = `https://epqs.nationalmap.gov/v1/json?x=${lon}&y=${lat}` +
                   `&wkid=4326&units=Feet&includeDate=false`;
      fetches.push(
        fetch(url)
          .then(r => r.json())
          .then(d => d.value)
          .catch(() => null)
      );
    }

    const elevations = await Promise.all(fetches);

    // place a green ground-clamped point there
    addSamplePoint(last_lat, last_lon);

    // 4) Plot, passing the callsign and endpoints
    plotProfile(elevations, {
      callsign,
      start: { lat: startLat, lon: startLon },
      end:   { lat: endLat,   lon: endLon   }
    }, 
      currentFeet, samples);
  });


// One moving hover marker on the globe
let hoverPointEntity = null;

function updateHoverPoint(lat, lon) {
  const position = Cesium.Cartesian3.fromDegrees(lon, lat, 0, Cesium.Ellipsoid.WGS84);
  if (!hoverPointEntity) {
    hoverPointEntity = viewer.entities.add({
      position,
      point: {
        pixelSize: 14,
        color: Cesium.Color.BLUE,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
      }
    });
  } else {
    hoverPointEntity.position = position;
  }
}
function clearHoverPoint() {
  if (hoverPointEntity) {
    viewer.entities.remove(hoverPointEntity);
    hoverPointEntity = null;
  }
}

  function addSamplePoint(lat, lon) {
    const position = Cesium.Cartesian3.fromDegrees(
      lon, lat, 0, Cesium.Ellipsoid.WGS84
    );
    viewer.entities.add({
      position,
      point: {
        pixelSize: 20,
        color: Cesium.Color.GREEN,
        heightReference: Cesium.HeightReference.CLAMP_TO_GROUND
      }
    });
  }

// Load a script once and cache the promise
const _scriptCache = new Map();
function loadScriptOnce(url) {
  if (_scriptCache.has(url)) return _scriptCache.get(url);
  const p = new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = url;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = (e) => reject(new Error(`Failed to load ${url}`));
    document.head.appendChild(s);
  });
  _scriptCache.set(url, p);
  return p;
}

// Ensure Chart.js is available (UMD builds expose global `Chart`)
function ensureChartJs() {
  if (window.Chart) return Promise.resolve();
  return loadScriptOnce("https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js");
}

async function plotProfile(elevations, { callsign, start, end }, feet, samples) {
   await ensureChartJs();
    const container = document.getElementById('profiles');
    const canvas    = document.createElement('canvas');
    canvas.className = 'profileCanvas';

    // give the canvas a stable id so we can find/scroll it later if needed
const id = `profile-${sanitizeId(callsign)}-${feet}`;
canvas.id = id;



    container.appendChild(canvas);


// register in index
profileIndex.set(keyFor(callsign, feet), { canvas, callsign, feet });


    const ctx    = canvas.getContext('2d');
    const labels = elevations.map((_, i) => (maxFeet / (elevations.length - 1)) * i);

const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Elevation (ft)',
          data: elevations,
          borderColor: 'blue',
          fill: false,
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            font: { size: 12 },
            text: [
              `Callsign: ${callsign}`,
              `Start: ${start.lat.toFixed(4)}, ${start.lon.toFixed(4)}`,
              `End:   ${end.lat.toFixed(4)}, ${end.lon.toFixed(4)}`
            ]
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Feet along path' }
          },
          y: {
            title: { display: true, text: 'Elevation (ft)' }
          }
        }
      }
    });

  // —— Hover handlers: move blue dot along the geodesic samples —— //
  canvas.addEventListener('mousemove', (evt) => {
    const elems = chart.getElementsAtEventForMode(evt, 'nearest', { intersect: false }, false);
    if (!elems || elems.length === 0) return;
    const idx = elems[0].index;
    // Guard for out-of-range or missing sample/elevation
    if (idx < 0 || idx >= samples.length) return;
    const elev = elevations[idx];
    if (elev == null) return; // skip gaps
    const { lat, lon } = samples[idx];
    updateHoverPoint(lat, lon);
  });

  canvas.addEventListener('mouseleave', () => {
    clearHoverPoint();
  });

 // Clicking the chart focuses the corresponding polyline in Cesium and moves the slider
 canvas.addEventListener('click', () => {
   const entity = findEntityById(callsign);
   if (!entity) {
     console.warn('No entity found for callsign:', callsign);
     return;
   }
   focusEntityTime(entity);
 });


}

