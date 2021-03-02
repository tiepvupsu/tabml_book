# add the following 2 lines into your .bashrc or .zhsrc (or your choice of bash shell):
# export TABML='/path/to/this/repo'
# alias 2tabml='cd $TABML; source tabml_env/bin/activate; source bashrc'
alias tabml_build='2tabml; jupyter-book build book/'
function tabml_deploy() {
  2tabml
  export DEPLOY='_deploy'
  rm -rf $DEPLOY
  mkdir $DEPLOY
  git clone --single-branch --branch gh-pages https://github.com/tiepvupsu/tabml_book $DEPLOY/
  cd $DEPLOY
  rm -Rf *
  cp -r ../book/_build/html/ ./
  git add -f --all .
  git commit -m ":rocket: Deploy date +\"%Y-%m-%d_%H-%M-%S\""
  git push
  cd ../
  rm -rf $DEPLOY
}

function convert_notebooks() {
# for every notebook in one folder:
#   convert it to markdown using jupytext
#   prefix it by nb_ to avoid multiple files with same filenames (even with diff exts)

for nb in $(ls *.ipynb)
do
  jupytext $nb --to myst
done

for nb in $(ls *.ipynb)
do
  mv $nb nb_$nb
done
ls
}

function gc_doc() {git commit -m ":memo: $1"}
