import lmdb
import six
import numpy as np
from PIL import Image

read_root_dir = '/data/zhui/back/NIPS2014'
write_root_dir = '/home/mkyang/data/sub_nips2014'
read_env = lmdb.open(read_root_dir, max_readers=32, readonly=True)
write_env = lmdb.open(write_root_dir, map_size=1099511627776)

def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k.encode(), v)

assert read_env is not None, "cannot create lmdb from %s" % read_root_dir
read_txn = read_env.begin()
nSamples = int(read_txn.get(b"num-samples"))
sub_nsamples = 10000
indices = list(np.random.permutation(nSamples))
indices = indices[:sub_nsamples]

cache = {}
for i, index in enumerate(indices):
  img_key = b'image-%09d' % index
  label_key = b'label-%09d' % index

  imgbuf = read_txn.get(img_key)
  word = read_txn.get(label_key)

  new_img_key = 'image-%09d' % (i+1)
  new_label_key = 'label-%09d' % (i+1)
  cache[new_img_key] = imgbuf
  cache[new_label_key] = word

cache['num-samples'] = str(sub_nsamples).encode()
writeCache(write_env, cache)