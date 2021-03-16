# from fias to cx1g8:
# copy only the experiment folder (only recent files)
# rsync -anvzr --delete --include-from=<(ssh wilmot@fias.uni-frankfurt.de 'find ~/Documents/code/aec-tf-v2/experiments/ -mtime -4 -type d -mindepth 1 -maxdepth 1 -printf "%P\n"') --exclude=* --delete-excluded wilmot@fias.uni-frankfurt.de:~/Documents/code/aec-tf-v2/experiments/ ~/Code/aec-tf-v2/experiments
rsync -anvzr --delete --include-from=<(ssh wilmot@fias.uni-frankfurt.de 'find ~/Documents/code/aec-tf-v2/experiments/ -mtime -4 -mindepth 1 -printf "%P\n"') --exclude=* --delete-excluded wilmot@fias.uni-frankfurt.de:~/Documents/code/aec-tf-v2/experiments/ ~/Code/aec-tf-v2/experiments

echo    # (optional) move to a new line
echo    # (optional) move to a new line
read -p "Do you want to proceed? " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi
# rsync -azPr --delete --include-from=<(ssh wilmot@fias.uni-frankfurt.de 'find ~/Documents/code/aec-tf-v2/experiments/ -mtime -4 -type d -mindepth 1 -maxdepth 1 -printf "%P\n"') --exclude=* --delete-excluded wilmot@fias.uni-frankfurt.de:~/Documents/code/aec-tf-v2/experiments/ ~/Code/aec-tf-v2/experiments
rsync -azPr --delete --include-from=<(ssh wilmot@fias.uni-frankfurt.de 'find ~/Documents/code/aec-tf-v2/experiments/ -mtime -4 -mindepth 1 -printf "%P\n"') --exclude=* --delete-excluded wilmot@fias.uni-frankfurt.de:~/Documents/code/aec-tf-v2/experiments/ ~/Code/aec-tf-v2/experiments
