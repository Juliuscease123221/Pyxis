// Zero-copy decoders for the tile and k-NN binaries.
//
// The tile payload is struct-of-arrays precisely so this file can hand back
// typed-array views over the received buffer without allocating or parsing
// anything. An interleaved layout would force a DataView loop here, which is
// the cost the binary format exists to avoid.

export const TILE_HEADER = 28;   // magic(4) version(2) flags(2) count(4) bbox(16)

export function decodeTile(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1),
                                    dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'ATLS') throw new Error(`bad tile magic: ${magic}`);
  const version = dv.getUint16(4, true);
  if (version !== 1) throw new Error(`unsupported tile version ${version}`);
  const flags = dv.getUint16(6, true);
  if (!(flags & 1)) throw new Error('interleaved tiles are not supported by this viewer');
  const count = dv.getUint32(8, true);
  const bbox = [dv.getFloat32(12, true), dv.getFloat32(16, true),
                dv.getFloat32(20, true), dv.getFloat32(24, true)];

  let o = TILE_HEADER;
  const x   = new Float32Array(buf, o, count); o += count * 4;
  const y   = new Float32Array(buf, o, count); o += count * 4;
  const id  = new Uint32Array(buf, o, count);  o += count * 4;
  const cat = new Uint32Array(buf, o, count);  o += count * 4;
  const imp = new Uint16Array(buf, o, count);
  return { count, bbox, x, y, id, cat, imp };
}

// knn.bin: magic(4) "AKNN" version(2) k(2) n(4), then n*k uint32 point indices.
export function decodeKnn(buf) {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1),
                                    dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'AKNN') throw new Error(`bad knn magic: ${magic}`);
  const k = dv.getUint16(6, true);
  const n = dv.getUint32(8, true);
  const flat = new Uint32Array(buf, 12, n * k);
  return { k, n, neighboursOf: (i) => flat.subarray(i * k, i * k + k) };
}
