{pkgs ? import <nixpkgs>{}}:
pkgs.mkShell {
  name = "cpp-env-shell";
  buildInputs = with pkgs; [
      autoconf binutils
      clang-tools cmake coreutils gcc gnumake stdenv.cc
      freeglut
      gbenchmark gtest gperf
      gnupg libGL libGLU zlib
      m4 utillinux
      pngpp procps tbb unzip
      openblas
      SDL2
      SDL2_ttf
  ];
  shellHook = ''
    #export EXTRA_CCFLAGS="-I/usr/include $EXTRA_CCFLAGS"
  '';
}
