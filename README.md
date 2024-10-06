# process_openwebtext
This is a python script that works with openwebtext, Reference to the processing script [nanoGPT] (https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py).
Before using this script, you need:
Manually download the openwebtext dataset from huggingface or hf-mirror. Then, unzip them all.
Folders are organized roughly as follows,
```python
-openwebtxt
--subsets
----xx.tar
...
--unzip_folder
----xxx.xz
...
--txt_folder
----xxx.txt
...
--merged_openwebtxt.json
```
As part of this script is to convert the unzipped xz file to a txt file, you can also download the converted txt file directly [here](https://mega.nz/folder/EZZD0YwJ#9_PlEQzdMVLaNdKv_ICNVQ/folder/cc4RgQQZ).

