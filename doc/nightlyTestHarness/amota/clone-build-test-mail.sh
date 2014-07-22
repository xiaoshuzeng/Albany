#!/bin/bash

source $HOME/.bashrc

keychain id_rsa
               [[ -f $HOME/.keychain/$HOSTNAME-sh ]] && \
                       source $HOME/.keychain/$HOSTNAME-sh

NUM_PROCS=`nproc`

LCM_DIR="/home/lcm"

# trilinos required before albany
PACKAGES="trilinos albany"
TOOL_CHAINS="gcc clang"
BUILD_TYPES="debug release"

cd "$LCM_DIR"

for PACKAGE in $PACKAGES; do
    case "$PACKAGE" in
	trilinos)
	    PACKAGE_NAME="Trilinos"
	    REPO="software.sandia.gov:/space/git/$PACKAGE_NAME"
	    ;;
	albany)
	    PACKAGE_NAME="Albany"
	    REPO="git@github.com:gahansen/Albany.git"
	    ;;
	*)
	    echo "Unrecognized package option"
	    exit 1
	    ;;
    esac
    PACKAGE_DIR="$LCM_DIR/$PACKAGE_NAME"
    CHECKOUT_LOG="$PACKAGE-checkout.log"
    if [ -d $PACKAGE_DIR ]; then
	rm $PACKAGE_DIR -rf
    fi
    git clone -v "$REPO" &> "$CHECKOUT_LOG"
done

COMMAND="$LCM_DIR/build-test-mail.sh"

for PACKAGE in $PACKAGES; do
    for TOOL_CHAIN in $TOOL_CHAINS; do
	for BUILD_TYPE in $BUILD_TYPES; do
	    "$COMMAND" "$PACKAGE" "$TOOL_CHAIN" "$BUILD_TYPE" "$NUM_PROCS" \
		"$LCM_DIR"
	done
    done
done

cd "$LCM_DIR"
