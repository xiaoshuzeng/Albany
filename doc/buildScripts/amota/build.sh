#!/bin/bash

SCRIPT_NAME=`basename $0`
PACKAGE=$1
TOOL_CHAIN=$2
BUILD_TYPE=$3
NUM_PROCS=$4
LCM_DIR=`pwd`

if [ -z "$PACKAGE" ]; then
    echo "Specifiy package [trilinos|albany]"
    exit 1
fi

if [ -z "$TOOL_CHAIN" ]; then
    echo "Specify tool chain [gcc|clang]"
    exit 1
fi

if [ -z "$BUILD_TYPE" ]; then
    echo "Specify build type [debug|release]"
    exit 1
fi

if [ -z "$NUM_PROCS" ]; then
    NUM_PROCS="1"
fi

case "$SCRIPT_NAME" in
    build.sh)
	;;
    clean.sh)
	;;
    clean-build.sh)
	;;
    *)
	echo "Unrecognized script name"
	exit 1
	;;
esac

case "$PACKAGE" in
    trilinos)
	PACKAGE_STRING="TRILINOS"
	PACKAGE_NAME="Trilinos"
	;;
    albany)
	PACKAGE_STRING="ALBANY"
	PACKAGE_NAME="Albany"
	;;
    *)
	echo "Unrecognized package option"
	exit 1
	;;
esac

CONFIG_FILE="config-$PACKAGE.sh"

case "$TOOL_CHAIN" in
    gcc)
	export OMPI_CC=`which gcc`
	export OMPI_CXX=`which g++`
	export OMPI_FC=`which gfortran`
	CMAKE_CXX_FLAGS="-ansi -Wall -pedantic -Wno-long-long"
	;;
    clang)
	export OMPI_CC=`which clang`
	export OMPI_CXX=`which clang++`
	export OMPI_FC=`which gfortran`
	CMAKE_CXX_FLAGS="-Weverything -pedantic -Wno-long-long -Wno-documentation"
	;;
    *)
	echo "Unrecognized tool chain option"
	exit 1
	;;
esac

case "$BUILD_TYPE" in
    debug)
	BUILD_STRING="DEBUG"
	;;
    release)
	BUILD_STRING="RELEASE"
	;;
    *)
	echo "Unrecognized build type option"
	exit 1
	;;
esac

BUILD=$TOOL_CHAIN-$BUILD_TYPE

PACKAGE_DIR="$LCM_DIR/$PACKAGE_NAME"
INSTALL_DIR="$LCM_DIR/install-$BUILD"
BUILD_DIR="$PACKAGE_DIR/build-$BUILD"

echo "------------------------------------------------------------"
echo -e "$PACKAGE_NAME directory\t: $PACKAGE_DIR"
echo -e "Install directory \t: $INSTALL_DIR"
echo -e "Build directory\t\t: $BUILD_DIR"
echo "------------------------------------------------------------"

cd "$LCM_DIR"

case "$SCRIPT_NAME" in
    build.sh)
	if [ ! -d "$BUILD_DIR" ]; then
	    echo "Build directory does not exist. Run:"
	    echo "  clean-build.sh $1 $2 $3 $4"
	    echo "to create."
	    exit 1
	fi
	echo "REBUILDING $PACKAGE_STRING ..."
	echo "------------------------------------------------------------"
	cd "$BUILD_DIR"
	;;
    clean.sh)
	case "$PACKAGE" in
	    trilinos)
		rm "$INSTALL_DIR" -rf
		;;
	    albany)
		;;
	    *)
		echo "Unrecognized package option"
		exit 1
		;;
	esac
	echo "CLEANING UP $PACKAGE_STRING ..."
	echo "------------------------------------------------------------"
	rm "$BUILD_DIR" -rf
	mkdir "$BUILD_DIR"
	cp -p "$CONFIG_FILE" "$BUILD_DIR"
	cd "$BUILD_DIR"
	sed -i -e "s|ompi_cc|$OMPI_CC|g;" "$CONFIG_FILE"
	sed -i -e "s|ompi_cxx|$OMPI_CXX|g;" "$CONFIG_FILE"
	sed -i -e "s|ompi_fc|$OMPI_FC|g;" "$CONFIG_FILE"
	sed -i -e "s|install_dir|$INSTALL_DIR|g;" "$CONFIG_FILE"
	sed -i -e "s|build_type|$BUILD_STRING|g;" "$CONFIG_FILE"
	sed -i -e "s|cmake_cxx_flags|$CMAKE_CXX_FLAGS|g;" "$CONFIG_FILE"
	./"$CONFIG_FILE"
	exit 0
	;;
    clean-build.sh)
	case "$PACKAGE" in
	    trilinos)
		rm "$INSTALL_DIR" -rf
		;;
	    albany)
		;;
	    *)
		echo "Unrecognized package option"
		exit 1
		;;
	esac
	echo "CLEANING UP AND REBUILDING $PACKAGE_STRING ..."
	echo "------------------------------------------------------------"
	rm "$BUILD_DIR" -rf
	mkdir "$BUILD_DIR"
	cp -p "$CONFIG_FILE" "$BUILD_DIR"
	cd "$BUILD_DIR"
	sed -i -e "s|ompi_cc|$OMPI_CC|g;" "$CONFIG_FILE"
	sed -i -e "s|ompi_cxx|$OMPI_CXX|g;" "$CONFIG_FILE"
	sed -i -e "s|ompi_fc|$OMPI_FC|g;" "$CONFIG_FILE"
	sed -i -e "s|install_dir|$INSTALL_DIR|g;" "$CONFIG_FILE"
	sed -i -e "s|build_type|$BUILD_STRING|g;" "$CONFIG_FILE"
	sed -i -e "s|cmake_cxx_flags|$CMAKE_CXX_FLAGS|g;" "$CONFIG_FILE"
	./"$CONFIG_FILE"
	echo "------------------------------------------------------------"
	;;
    *)
	echo "Unrecognized script name"
	exit 1
	;;
esac

ERROR_LOG="$LCM_DIR"/"$PACKAGE"-"$TOOL_CHAIN"-"$BUILD_TYPE"-error.log
echo "WARNINGS AND ERRORS REDIRECTED TO $ERROR_LOG"
echo "------------------------------------------------------------"
if [ -f "$ERROR_LOG" ]; then
    rm "$ERROR_LOG"
fi

case "$PACKAGE" in
    trilinos)
	make -j $NUM_PROCS 2> "$ERROR_LOG"
	STATUS=$?
	if [ $STATUS -ne 0 ]; then
	    echo "*** MAKE COMMAND FAILED ***"
	else
	    make install
	    STATUS=$?
	    if [ $STATUS -ne 0 ]; then
		echo "*** MAKE INSTALL COMMAND FAILED ***"
	    fi
	fi
	;;
    albany)
	make -j $NUM_PROCS 2> "$ERROR_LOG"
	STATUS=$?
	if [ $STATUS -ne 0 ]; then
	    echo "*** MAKE COMMAND FAILED ***"
	fi
	;;
    *)
	echo "Unrecognized package option"
	exit 1
	;;
esac
cd "$LCM_DIR"
