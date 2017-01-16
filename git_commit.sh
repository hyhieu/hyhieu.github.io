#!/bin/bash

git add --all .
git commit -m "$@"

git pull origin master

git push origin master
