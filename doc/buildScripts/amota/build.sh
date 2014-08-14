#!/bin/bash

source ./env-single.sh

cd "$LCM_DIR"

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

echo "------------------------------------------------------------"
echo -e "$PACKAGE_NAME directory\t: $PACKAGE_DIR"
echo -e "Install directory \t: $INSTALL_DIR"
echo -e "Build directory\t\t: $BUILD_DIR"
echo "------------------------------------------------------------"

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
	sed -i -e "s|package_dir|$PACKAGE_DIR|g;" "$CONFIG_FILE"
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
	sed -i -e "s|package_dir|$PACKAGE_DIR|g;" "$CONFIG_FILE"
	./"$CONFIG_FILE"
	echo "------------------------------------------------------------"
	;;
    *)
	echo "Unrecognized script name"
	exit 1
	;;
esac

ERROR_LOG="$LCM_DIR/$PREFIX-error.log"
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
