#!/bin/bash

PACKAGE=$1
TOOL_CHAIN=$2
BUILD_TYPE=$3
NUM_PROCS=$4
LCM_DIR=$5

PREFIX="$PACKAGE-$TOOL_CHAIN-$BUILD_TYPE"

BUILD_LOG="$PREFIX-build.log"

case "$PACKAGE" in
    trilinos)
	PACKAGE_NAME="Trilinos"
	;;
    albany)
	PACKAGE_NAME="Albany"
	;;
    *)
	echo "Unrecognized package option"
	exit 1
	;;
esac

cd "$LCM_DIR"

COMMAND="$LCM_DIR/clean-build.sh"
"$COMMAND" "$PACKAGE" "$TOOL_CHAIN" "$BUILD_TYPE" "$NUM_PROCS" &> "$BUILD_LOG"

HOST=`hostname`

case "$PACKAGE" in
    trilinos)
	;;
    albany)
	TEST_LOG="$LCM_DIR/$PREFIX-test.log"
	PACKAGE_DIR="$LCM_DIR/$PACKAGE_NAME"
	BUILD_DIR="$PACKAGE_DIR/build-$TOOL_CHAIN-$BUILD_TYPE"
	cd "$BUILD_DIR"
	ctest --timeout 300 . &> "$TEST_LOG"
	FROM="amota@sandia.gov"
	TO="albany-regression@software.sandia.gov"
	SUCCESS_RATE=`grep "tests failed" "$TEST_LOG"`
	HEADER="LCM TESTS: $HOST, $TOOL_CHAIN $BUILD_TYPE, $SUCCESS_RATE"
	mail -r "$FROM" -s "$HEADER" "$TO" < "$TEST_LOG"
	;;
    *)
	echo "Unrecognized package option"
	exit 1
	;;
esac

cd "$LCM_DIR"
