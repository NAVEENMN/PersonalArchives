#!/bin/bash
# Usage: bash ./setup.sh

setup_path=/tmp/setup.py

install_dependencies() {
  pip3 install clint
  # curl and get bltbot.py
}

dump_blob() {
  blob="#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import with_statement

import sys
import os

sys.path.insert(0, os.path.abspath('..'))

from clint.textui import puts, indent, colored

if __name__ == '__main__':
    with indent(4, quote=colored.blue('blt >> ')):
        puts('Welcome to BltBot, your system is now setup')
        puts('You can now run ')
        puts('python3 bltbot.py')
        puts('to invoke the bot')"
  echo "$blob" > "$setup_path"
}

setup() {
  # check if python3 is installed
  if command -v python3 &>/dev/null; then
    if [ "$(uname)" == "Linux" ]; then
      sudo apt-get update
      echo y | sudo apt-get install python3-distutils
      echo y | apt install python3-pip
    fi
    pip3 --version
    # check if pip33 is installed
    if [ $? -eq 0 ]; then
      install_dependencies
      dump_blob
      python3 /tmp/setup.py
      rm /tmp/setup.py
    else
      echo Please install python3 pip to proceed
      exit
    fi
  else
    echo "Please install python3 to proceed"
  fi
}

setup
