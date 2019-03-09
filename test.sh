#! /bin/bash

cd src
python -c \
"from handler import handler; x = handler({
    'urls': ['https://catzone-tcwebsites.netdna-ssl.com/wp-content/uploads/2017/01/tabby-cat-names.jpg']}, 
    None); print(x)"