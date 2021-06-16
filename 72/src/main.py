import time, datetime
start = time.time()

import multi_base_classifiers, multi_base_classifiers_2,\
  multi_base_classifiers_3, svms, random_forests, random_forests_2

print('run time, ', str(datetime.timedelta(seconds=(time.time() - start))), ', seconds', '\a')
