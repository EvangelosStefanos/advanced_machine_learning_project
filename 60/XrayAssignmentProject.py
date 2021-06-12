import sys

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, ADASYN
from imblearn.under_sampling import TomekLinks

import pipeline


#Main
args = sys.argv

if args.__contains__("-help"):
    print("USAGE: You can use any one of the following params to use the matching imbalance resolution methods:")
    print("\"-smote\"\n-\"borderlineSmote\"\n\"-svmSmote\"\n\"-KMeansSmote\"\n\"-SmoteTomek\"\n\"-adasyn\"")
    print()
    print("You can select which of the data-set case you use based on (as an additional optional param) :")
    print("\"-full\" (DEFAULT), which means that the full data-set is to be used.")
    print("\"-imba\", which means that an artificially imbalanced data-set is used.")
    print("\"-imbaVP\", which means that an artificially imbalanced + set xray 'View Position' data-set is used.")
    exit()


imbalance_method = None
if args.__contains__("-smote"):
    imbalance_method = SMOTE(random_state=42)
elif args.__contains__("-borderlineSmote"):
    imbalance_method = BorderlineSMOTE(random_state=42)
elif args.__contains__("-svmSmote"):
    imbalance_method = SVMSMOTE(random_state=42) # Warning SUPER SLOW
elif args.__contains__("-KMeansSmote"):
    imbalance_method = KMeansSMOTE(random_state=42)
elif args.__contains__("-SmoteTomek"):
    imbalance_method = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=42)
elif args.__contains__("-adasyn"):
    imbalance_method = ADASYN(random_state=42)

imba = args.__contains__("-imba")
imbaVP = args.__contains__("-imbaVP")
dataset_case = "'full'"
if imba:
    dataset_case = "'artificially imbalanced'"
elif imbaVP:
    dataset_case = "'artificially imbalanced + set View position'"


print("Running with", imbalance_method, "as the imbalance method and using the", dataset_case, "data-set.")
pipeline.run(imbalance_method, imba, imbaVP)