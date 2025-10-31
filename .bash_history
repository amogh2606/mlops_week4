ls
unzip archive\ 2.zip 
unzip "archive 2.zip" 
ls
rm archive\ 2.zip 
ls
tar -xvzf upload.tar.gz 
rm -rf archive\ 3/
tar -xvzf archive.tar.gz 
rm upload.tar.gz 
cd down/
cp -R * ..
cd ..
ls
tar -xf upload.tar
rm upload.tar
rm upload.tar.gz 
rm archive.tar.gz 
ls
clear
pip install dvc
pip install dvc[gs]
git init
dvc init
export DVC_BUCKET="gs://week-2-mlops-bucket/dvc_remote"
rm -rf archive\ 3/
rm -rf down/
rm -rf iris_classifier/
rm -rf dvc_remote/
mkdir artifacts
dvc remote add -d gcs_bucket DVC_BUCKET
cd data
ls
cd ..
cp-R data data2
cp -R data data2
ls
cd data2
ls
cd ..
cd data
rm -rf *
ls
cd ..
cp data2/v1/iris_data.csv data/V1_data.csv
cp data2/v2/iris_data_v2.csv data/V2_data.csv
cp data2/RAW/iris.csv data/V0_RAW.csv
dvc add data/V0_RAW.csv 
dvc add data/V1_data.csv 
dvc add data/V2_data.csv 
git add data/*.dvc
git commit -m "DVC:Added raw,v1 and v2 version data"
git config --global user.email "22f1001411@ds.study.iitm.ac.in"
git config --global user.name "amogh2606"
git commit -m "DVC:Added raw,v1 and v2 version data"
dvc push
ls
python training_pipeline.py 
python training_pipeline.py V0_RUN1
python training_pipeline.py data/V0_RAW.csv V0_RUN1
dvc add artifacts
git add artifacts.dvc
git commit -m "Model V1 : trained on raw data"
git tar model-v1
git tag model-v1
dvc push
python augment_data.py 
ls
dvc add data/V3_augmented.csv 
git add data/V3_augmented.csv.dvc 
dvc push
python training_pipeline.py data/V3_augmented.csv V3_AUGMENTED_RUN
dvc add artifacts
git add artifacts.dvc 
git commit -m "Model trinaed on augmented data"
git tar model-v3
git tag model-v3
dvc push
python training_pipeline.py data/V1_data.csv V1_RUN
dvc add artifacts
git add artifact
git add artifacts
git add -f artifacts
git tag model-v1
git tag model-v2
clear
git status
git add augment_data.py training_pipeline.py 
git commit
git commit -m "Added files"
clear
git status
rm upload.tar 
clear
git status
git log
git config user.name
git config user.email
git branch -m master main
git remote add origin https://github.com/amogh2606/mlops_week4.git
git push -u origin main
git push -u origin main
echo -e "pandas\nscikit-learn\npytest\ngcsfs\ndvc\ncml" > requirements.txt
mkdir test
git checkout -b dev
git push -u origin dev
mkdir .github/workflows
mkdir .github/workflows/
mkdir .github
cd .github/
mkdir workflows
vi ci.yaml
cd 
cp test tests
cp -r test tests
git add requirements.txt tests/ .github/workflows/yml
git add requirements.txt tests/ .github/workflows/yaml
git add requirements.txt tests/ .github/workflows/ci.yaml
rm .gitignore 
cd .github/workflows/
ls
vi ci.yaml
ls
cd ..
cd ..
git add requirements.txt tests/ .github/workflows/ci.yaml 
git commit -m "Setup unit test , CI pipepine, etc"
git push origin dev
git remotve -v
git remote -v
git checkout main
git pull origin main
git merge dev
git push origin main
git branch -d dev
git push origin --delete dev
git checkout -b dev
git push -u origin dev
cd .github/
ls
cd workflows
ls
open ci.yaml 
vi ci.yaml 
cd ..
ls
cp ../requirements.txt .
ls
cd ..
git add .github/requirements.txt .github/workflows/ci.yaml 
git add tests/
git commit -m "Set up unit test,CI pipeline and Reporting"
git push origin dev
cd .github/workflows/
vi ci.yaml 
cd 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
vi .github/workflows/ci.yaml 
vi .github/requirements.txt 
git add .github/requirements.txt .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
pip freeze > requirements.txt
pip freeze
pip freeze > requirements.txt
vi requirements.txt 
vi .github/workflows/ci.yaml 
vi .github/workflows/ci.yaml 
vi .github/requirements.txt 
git add .github/requirements.txt .github/workflows/ci.yaml 
git push origin dev
git commit -m "fix"
git push origin dev
    copying numpy/tests/__init__.py -> build/lib.linux-x86_64-3.10/numpy/tests
            copying numpy/tests/test_reloading.py -> build/lib.linux-x86_64-3.10/numpy/tests
            copying numpy/tests/test_ctypeslib.py -> build/lib.linux-x86_64-3.10/numpy/tests
            copying numpy/tests/test_matlib.py -> build/lib.linux-x86_64-3.10/numpy/tests
            copying numpy/tests/test_public_api.py -> build/lib.linux-x86_64-3.10/numpy/tests
            copying numpy/tests/test_warnings.py -> build/lib.linux-x86_64-3.10/numpy/tests
            running build_clib
            customize UnixCCompiler
            customize UnixCCompiler using new_build_clib
            building 'npymath' library
            compiling C sources
            C compiler: gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC
      
            creating build/temp.linux-x86_64-3.10
            creating build/temp.linux-x86_64-3.10/numpy
            creating build/temp.linux-x86_64-3.10/numpy/core
            creating build/temp.linux-x86_64-3.10/numpy/core/src
            creating build/temp.linux-x86_64-3.10/numpy/core/src/npymath
            creating build/temp.linux-x86_64-3.10/build
            creating build/temp.linux-x86_64-3.10/build/src.linux-x86_64-3.10
            creating build/temp.linux-x86_64-3.10/build/src.linux-x86_64-3.10/numpy
            creating build/temp.linux-x86_64-3.10/build/src.linux-x86_64-3.10/numpy/core
            creating build/temp.linux-x86_64-3.10/build/src.linux-x86_64-3.10/numpy/core/src
            creating build/temp.linux-x86_64-3.10/build/src.linux-x86_64-3.10/numpy/core/src/npymath
            compile options: '-Ibuild/src.linux-x86_64-3.10/numpy/core/src/npymath -Inumpy/core/include -Ibuild/src.linux-x86_64-3.10/numpy/core/include/numpy -Inumpy/core/src/common -Inumpy/core/src -Inumpy/core -Inumpy/core/src/npymath -Inumpy/core/src/multiarray -Inumpy/core/src/umath -Inumpy/core/src/npysort -I/opt/hostedtoolcache/Python/3.10.19/x64/include/python3.10 -Ibuild/src.linux-x86_64-3.10/numpy/core/src/common -Ibuild/src.linux-x86_64-3.10/numpy/core/src/npymath -c'
            extra options: '-std=c99'
            gcc: numpy/core/src/npymath/npy_math.c
            gcc: numpy/core/src/npymath/halffloat.c
            gcc: build/src.linux-x86_64-3.10/numpy/core/src/npymath/npy_math_complex.c
            gcc: numpy/core/src/npymath/halffloat.c
            gcc: numpy/core/src/common/array_assign.c
            gcc: numpy/core/src/common/mem_overlap.c
            gcc: numpy/core/src/umath/extobj.c
            gcc: build/src.linux-x86_64-3.10/numpy/core/src/umath/scalarmath.c
            gcc: numpy/core/src/common/npy_longdouble.c
            gcc: numpy/core/src/common/ucsnarrow.c
            gcc: numpy/core/src/common/ufunc_override.c
            gcc: numpy/core/src/common/numpyos.c
            gcc: build/src.linux-x86_64-3.10/numpy/core/src/common/npy_cpu_features.c
            gcc: numpy/core/src/multiarray/mapping.c
            gcc: numpy/core/src/multiarray/methods.c
            gcc: numpy/core/src/umath/ufunc_type_resolution.c
            gcc: build/src.linux-x86_64-3.10/numpy/core/src/umath/matmul.c
            error: Command "gcc -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -fPIC -DNPY_INTERNAL_BUILD=1 -DHAVE_NPY_CONFIG_H=1 -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE=1 -D_LARGEFILE64_SOURCE=1 -Ibuild/src.linux-x86_64-3.10/numpy/core/src/umath -Ibuild/src.linux-x86_64-3.10/numpy/core/src/npymath -Ibuild/src.linux-x86_64-3.10/numpy/core/src/common -Inumpy/core/include -Ibuild/src.linux-x86_64-3.10/numpy/core/include/numpy -Inumpy/core/src/common -Inumpy/core/src -Inumpy/core -Inumpy/core/src/npymath -Inumpy/core/src/multiarray -Inumpy/core/src/umath -Inumpy/core/src/npysort -I/opt/hostedtoolcache/Python/3.10.19/x64/include/python3.10 -Ibuild/src.linux-x86_64-3.10/numpy/core/src/common -Ibuild/src.linux-x86_64-3.10/numpy/core/src/npymath -c build/src.linux-x86_64-3.10/numpy/core/src/multiarray/scalartypes.c -o build/temp.linux-x86_64-3.10/build/src.linux-x86_64-3.10/numpy/core/src/multiarray/scalartypes.o -MMD -MF build/temp.linux-x86_64-3.10/build/src.linux-x86_64-3.10/numpy/core/src/multiarray/scalartypes.o.d -std=c99" failed with exit status 1
            [end of output]
      
        note: This error originates from a subprocess, and is likely not a problem with pip.
        ERROR: Failed building wheel for numpy
      Failed to build numpy
      error: failed-wheel-build-for-install
      
      × Failed to build installable wheels for some pyproject.toml based projects
      ╰─> numpy
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error
× pip subprocess to install build dependencies did not run successfully.
│ exit code: 1
╰─> See above for output.
note: This error originates from a subprocess, and is likely not a problem with pip.
Error: Process completed with exit code 1.
clear
git push origin dev
git branch
git add data/V1_data.csv
git -f add data/V1_data.csv
git add -f data/V1_data.csv
git commit -m "added v1 data"
git push origin dev
git pull origin dev
git add requirements.txt
git commit -m "FIX: Updated package versions to resolve dependency errors from CI"
git pull origin dev
git pull origin dev
git push origin dev
cd data
ls
cd ..
git add data/V2_data.csv
git add -f data/V2_data.csv
git add -f data/V3_augmented.csv
git commit -m "add data"
git push origin dev
vi .github/workflows/ci.yaml 
dvc push
dvc remote modify gcs_bucket url gs://week-2-mlops-bucket/dvc_remote
dvc push
git puch origin dev
git push origin dev
vi .github/workflows/ci.yaml 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
git push origin dev
dvc add artifacts
git rm -r --cached 'artifacts'
dvc add artifacts
git add artifacts.dvc
git commit -m "FIX: Re-versioned model artifacts to ensure cache files are correctly tracked."
dvc push
git push origin dev
git add requirements.txt 
git commit -m "requirements fix"
git push origin dev
git add requirements.txt 
git commit -m "requirements fix"
git push origin dev
vi .github/workflows/ci.yaml 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
git add .github/workflows/ci.yaml 
vi .github/workflows/ci.yaml 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
git push origin dev
git add .github/workflows/ci.yaml 
vi .github/workflows/ci.yaml 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
git add .github/workflows/ci.yaml 
vi .github/workflows/ci.yaml 
git add requirements.txt .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
vi .github/workflows/ci.yaml 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
vi .github/workflows/ci.yaml 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
vi .github/workflows/ci.yaml 
git add .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
vi .github/workflows/ci.yaml 
git add requirements.txt .github/workflows/ci.yaml 
git commit -m "fix"
git push origin dev
git add requirements.txt 
git commit -m "fix"
git commit -m "fix"
git add requirements.txt 
git commit -m "fix"
git add requirements.txt 
git commit -m "fix"
git pull origin dev
git push orign dev
git pull dev
pip install cml
npm install -g @iterative/cml
tar -xvzf archive.tar.gz .
tar -cvzf archive_week4.tar.gz .
