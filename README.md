# Human-in-the-loop-TUM
This repository based on the TREC dataset implement different augmentation method including the language level as well as the embedding level and user can simply pick the data they want to augment and check the model performance.
In addition, this is the project outcome of TUM's XAL practice course. The project report can be found at [XAI_Report](/XAI_Report.pdf).

# Usage
* Easy data augmentation methods are implemented in [eda.py](eda.py) and [eda.ipynb](eda.ipynb).
* Back translation augmentation method is implemented in [back_translation.py](back_translation.py) and [back_translation.ipynb](back_translation.ipynb), where the German-English and Russian-English back translations are used.
* The embedding level data augmentation method is implemented in [augmentation_all_methods.ipynb](augmentation_all_methods.ipynb). Here we also encapsulate the user's discretion to choose how the three methods are combined and the process for augmenting data or data groups.

# Data
The [data](data) folder contains the augmentation data generated by the three augmentation methods. This data can also be generated by running the corresponding .ipynb file.

# Reference
The implementation of the eda and back translation methods is referenced from [DoubleMix](https://github.com/declare-lab/DoubleMix).

# Contact
If there are any questions, welcome to contact us by [ge47pez@mytum.de](ge47pez@mytum.de).
