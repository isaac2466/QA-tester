// Persistencia en IndexedDB: guarda imágenes (Blob/File) y resultados para que
// sobrevivan a un F5. localStorage no sirve aquí: tiene ~5 MB y sólo guarda texto.
const DB = "qa-tester";
const STORE = "state";

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function tx(mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => resolve(req?.result);
    t.onerror = () => reject(t.error);
  });
}

export async function load(key) {
  try {
    return await tx("readonly", (s) => s.get(key));
  } catch {
    return undefined; // modo privado o almacenamiento bloqueado: seguimos sin persistencia
  }
}

export async function save(key, value) {
  try {
    await tx("readwrite", (s) => (value == null ? s.delete(key) : s.put(value, key)));
  } catch {
    /* sin persistencia, la app sigue funcionando */
  }
}
