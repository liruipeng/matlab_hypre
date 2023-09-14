# matlab_hypre
matlab interface for hypre

A minor modification might be necessary on MacOS after compiling:
```console
$ cd mex
$ install_name_tool -change @rpath/libHYPRE-2.29.0.dylib ../hypre/lib/libHYPRE.dylib hypre_amg_setup.mexmaci64
```
