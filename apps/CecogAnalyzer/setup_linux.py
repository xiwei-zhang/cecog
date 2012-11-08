# -*- coding: utf-8 -*-
"""
                           The CellCognition Project
                     Copyright (c) 2006 - 2010 Michael Held
                      Gerlich Lab, ETH Zurich, Switzerland
                              www.cellcognition.org

              CellCognition is distributed under the LGPL License.
                        See trunk/LICENSE.txt for details.
                 See trunk/AUTHORS.txt for author contributions.

setup_linux - linux specific instruction for distutils setup
"""

__author__ = 'rudolf.hoefler@gmail.com'

import os
import sys
import shutil
import pkginfo

from datafiles import get_data_files, get_include_files

if not sys.platform.startswith("linux"):
    raise RuntimeError("%s runs only on Linux machine's"
                       % os.path.basename(__file__))
try:
    from cx_Freeze import setup, Executable
except ImportError:
    raise ImportError(("Could not import cx_Freeze.\n"
                       "Setting PYTHONPATH may help"))

PACKAGES = [ 'cecog', 'h5py', 'vigra', 'matplotlib' ]

INCLUDES = [ 'sip',
             'joblib',
             # if scipy-version >= 0.11
             # 'scipy.sparse.csgraph._validation',
             # 'scipy.sparse.csgraph._shortest_path',
             'scipy.spatial.kdtree']

EXCLUDES = [ 'PyQt4.QtDesigner', 'PyQt4.QtNetwork',
             'PyQt4.QtOpenGL', 'PyQt4.QtScript',
             'PyQt4.QtSql', 'PyQt4.QtTest',
             'PyQt4.QtWebKit', 'PyQt4.QtXml',
             'PyQt4.phonon',
             'rpy',
             '_gtkagg', '_tkagg', '_agg2', '_cairo', '_cocoaagg',
             '_fltkagg', '_gtk', '_gtkcairo',
             'Tkconstants', 'Tkinter', 'tcl' ]

# DLL_EXCLUDES = [ 'libgdk-win32-2.0-0.dll',
#                  'libgobject-2.0-0.dll',
#                  'libgdk_pixbuf-2.0-0.dll',
#                  'w9xpopen.exe' ] # is not excluded for some reasion


setup( options  ={ "build_exe": { "include_files": get_include_files(),
                                  "includes": INCLUDES,
                                  "excludes": EXCLUDES,
                                  "packages": PACKAGES,
                                  "optimize": 1,
                                 },
                   "install": {"prefix": "dist"} },
       executables = [Executable('CecogAnalyzer.py')],
       **pkginfo.metadata
)
