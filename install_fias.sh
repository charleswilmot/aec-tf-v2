rm -rf /tmp/test_script
mkdir /tmp/test_script
touch /tmp/test_script/.bashrc
install_root=/tmp/test_script


echo "----- Cloning the repository into $install_root/Code/aec-tf-v2/"
mkdir -p $install_root/Code
cd $install_root/Code
git clone https://github.com/charleswilmot/aec-tf-v2.git --quiet


echo "----- Installing all third-party python libraries"
cd aec-tf-v2
pip install -r requirements.txt


echo "----- Installing CoppeliaSim locally into $install_root/Softwares"
mkdir -p $install_root/Softwares
cd $install_root/Softwares

ubuntu_version=`lsb_release -rs`

if [[ $ubuntu_version = 20.04 ]]; then
  #statements
  echo "----- Detected Ubuntu version 20.04, downloading ~200Mb, this might take a few minutes"
  wget -q --show-progress --no-check-certificate -c https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_2_0_Ubuntu20_04.tar.xz -O - | tar -xJ
fi

if [[ $ubuntu_version = 18.04 ]]; then
  #statements
  echo "----- Detected Ubuntu version 18.04, downloading ~200Mb, this might take a few minutes"
  wget -q --show-progress --no-check-certificate -c https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_2_0_Ubuntu18_04.tar.xz -0 - | tar -xJ
fi
echo "----- CoppeliaSim installed"

echo "----- Cleaning .bashrc from potential previous changes"
perl -i -p0e 's/\n\n# PyRep environment variables.*# PyRep environment variables \(END\)\n//s' $install_root/.bashrc

echo "----- Setting environemnt variables in the $install_root/.bashrc file"
echo "

# PyRep environment variables
export COPPELIASIM_ROOT=$install_root/Softwares/CoppeliaSim_Edu_V4_2_0_Ubuntu${ubuntu_version}/
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=\$COPPELIASIM_ROOT
export COPPELIASIM_MODEL_PATH=$install_root/Code/aec-tf-v2/models/
# PyRep environment variables (END)
" >> $install_root/.bashrc

echo "----- Sourcing the $install_root/.bashrc file"
source $install_root/.bashrc
