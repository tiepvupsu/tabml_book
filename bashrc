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
