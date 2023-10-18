#!/bin/bash



# python attack.py -sv -n 10 -c  -e 100 -e1 150  -w  -c  -m efficientnet -mw CentralizedEfficientNetCIFAR10 -s efficientnet # add rest of your arguments
# python attack.py -sv -n 10 -c  -e 100 -e1 150  -w  -c  -m efficientnet -mw Federated2efficientnetCIFAR10 -s efficientnet # add rest of your arguments
# python attack.py -sv -n 10 -c  -e 100 -e1 150  -w  -c  -m efficientnet -mw Federated3efficientnetCIFAR10 -s efficientnet # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated5efficientnetCIFAR10 -s efficientnet # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated10efficientnetCIFAR10 -s efficientnet # add rest of your arguments

python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw CentralizedEfficientNetCIFAR100 -s efficientnet  -d CIFAR100 # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated2efficientnetCIFAR100 -s efficientnet  -d CIFAR100 # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated3efficientnetCIFAR100 -s efficientnet  -d CIFAR100 # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated5efficientnetCIFAR100 -s efficientnet  -d CIFAR100 # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated10efficientnetCIFAR100 -s efficientnet  -d CIFAR100 # add rest of your arguments


python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw CentralizedEfficientNetSVHN -s efficientnet  -d SVHN # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated2efficientnetSVHN -s efficientnet  -d SVHN # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated3efficientnetSVHN -s efficientnet  -d SVHN # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated5efficientnetSVHN -s efficientnet  -d SVHN # add rest of your arguments
python attack.py -sv -n 10 -c  -e 100 -e1 150  -w    -m efficientnet -mw Federated10efficientnetSVHN -s efficientnet  -d SVHN # add rest of your arguments