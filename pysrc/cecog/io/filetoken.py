"""
                          The CellCognition Project
                  Copyright (c) 2006 - 2009 Michael Held
                   Gerlich Lab, ETH Zurich, Switzerland

           CellCognition is distributed under the LGPL License.
                     See trunk/LICENSE.txt for details.
               See trunk/AUTHORS.txt for author contributions.
"""

__docformat__ = "epytext"
__author__ = 'Michael Held'
__date__ = '$Date$'
__revision__ = '$Rev$'
__source__ = '$URL::                                                          $'

__all__ = ['FileTokenImporter',
           'MetaMorphTokenImporter',
           'SimpleTokenImporter',
           'ZeissLifeTokenImporter',
           ]

#------------------------------------------------------------------------------
# standard library imports:
#
import os

#------------------------------------------------------------------------------
# extension module imports:
#
from pdk.fileutils import collect_files
from pdk.iterator import unique
from cecog import ccore

#------------------------------------------------------------------------------
# cecog imports:
#
from cecog.util.token import (Token,
                              TokenHandler)
from cecog.io.imagecontainer import (MetaData,
                                     DIMENSION_NAME_POSITION,
                                     DIMENSION_NAME_TIME,
                                     DIMENSION_NAME_CHANNEL,
                                     DIMENSION_NAME_ZSLICE,
                                     META_INFO_WELL,
                                     META_INFO_SUBWELL
                                     )

#------------------------------------------------------------------------------
# constants:
#

#------------------------------------------------------------------------------
# functions:
#


#------------------------------------------------------------------------------
# classes:
#
class DefaultCoordinates(object):
    def __init__(self):
        self.default_values = {
                               DIMENSION_NAME_TIME: '1',
                               DIMENSION_NAME_CHANNEL: 'ch0',
                               DIMENSION_NAME_ZSLICE: '1',
                               'DEFAULT': ''
                               }
        return

    def __call__(self, image_info, key):
        if self.default_values.has_key(key):
            if not image_info.has_key(key):
                return self.default_values[key]
            elif image_info[key] in ['', None]:
                return self.default_values[key]
        else:
            if image_info.has_key(key):
                return image_info[key]
            else:
                return self.default_values['DEFAULT']

        return image_info[key]

class FileTokenImporter(object):

    EXTENSIONS = ['.tif', '.png', '.png']
    IGNORE_PREFIXES = ['.']

    MULTIIMAGE_IGNORE = 'ignore'
    MULTIIMAGE_USE_ZSLICE = 'zslice'

    def __init__(self, path, token_handler,
                 extensions=None, ignore_prefixes=None, multi_image=None):
        self.path = os.path.normpath(path)
        self.extensions = self.EXTENSIONS if extensions is None \
                          else extensions
        self.ignore_prefixes = self.IGNORE_PREFIXES if ignore_prefixes is None \
                               else ignore_prefixes
        self.multi_image = self.MULTIIMAGE_USE_ZSLICE if multi_image is None \
                           else multi_image
        self.has_multi_images = False
        self.token_handler = token_handler
        self.meta_data = MetaData()
        self.dimension_lookup = self._build_dimension_lookup()
        self.meta_data.setup()
        #print self.meta_data
        #print self.dimension_lookup

    def _build_token_list(self):
        file_list = collect_files(self.path, self.extensions, absolute=True,
                                  follow=False, recursive=True,
                                  ignore_case=True, force_python=True)

        token_list = []
        for filename in file_list:
            filename_rel = filename[len(self.path)+1:]
            filename_short = os.path.split(filename_rel)[1]
            if filename_short[0] not in self.ignore_prefixes:
                result = self.token_handler.search_all(filename_short)
                result['filename'] = filename_rel
                result['timestamp'] = os.path.getmtime(filename)
                token_list.append(result)
        return token_list


    def _build_dimension_lookup(self):
        token_list = self._build_token_list()
        lookup = {}
        has_xy = False

        positions = []
        times = []
        channels = []
        zslices = []

        for item in token_list:

            if not has_xy:
                has_xy = True
                info = ccore.ImageImportInfo(os.path.join(self.path,
                                                          item['filename']))
                self.meta_data.dim_x = info.width
                self.meta_data.dim_y = info.height
                self.meta_data.pixel_type = info.pixel_type
                self.has_multi_images = info.images > 1

            position = item[DIMENSION_NAME_POSITION]
            if not position in lookup:
                lookup[position] = {}
            time = item[DIMENSION_NAME_TIME]
            if not time in lookup[position]:
                lookup[position][time] = {}
            channel = item[DIMENSION_NAME_CHANNEL]
            if not channel in lookup[position][time]:
                lookup[position][time][channel] = {}
            zslice = item[DIMENSION_NAME_ZSLICE]
            if not zslice in lookup[position][time][channel]:
                lookup[position][time][channel][zslice] = item['filename']

            self.meta_data.append_absolute_time(position,
                                                time,
                                                item['timestamp'])

            if (self.has_multi_images and
                self.multi_image == self.MULTIIMAGE_USE_ZSLICE):
                if not zslice is None:
                    raise ValueError('Multi-image assigned for zslice conflicts'
                                     ' with zslice token in filename!')
                zslices.extend(range(1,info.images+1))
            else:
                zslices.append(zslice)

            positions.append(position)
            times.append(time)
            channels.append(channel)

        self.meta_data.positions = tuple(sorted(unique(positions)))
        self.meta_data.times = tuple(sorted(unique(times)))
        self.meta_data.channels = tuple(sorted(unique(channels)))
        self.meta_data.zslices = tuple(sorted(unique(zslices)))
        return lookup

    def get_image(self, position, frame, channel, zslice):
        index = 0
        if (self.has_multi_images and
            self.multi_image == self.MULTIIMAGE_USE_ZSLICE):
            index = zslice - 1
            zslice = None
        #print position, frame, channel, zslice, index
        filename_rel = self.dimension_lookup[position][frame][channel][zslice]
        filename_abs = os.path.join(self.path, filename_rel)
        image = ccore.readImage(filename_abs, image_index=index)
        return image


class SimpleTokenImporter(FileTokenImporter):

    TOKEN_P = None
    TOKEN_T = None
    TOKEN_C = None
    TOKEN_Z = None

    def __init__(self, path, separator='_',
                 extensions=None, ignore_prefixes=None, multi_image=None):
        simple_token = TokenHandler(separator=separator)
        simple_token.register_token(self.TOKEN_P)
        simple_token.register_token(self.TOKEN_T)
        simple_token.register_token(self.TOKEN_C)
        simple_token.register_token(self.TOKEN_Z)

        super(SimpleTokenImporter,
              self).__init__(path, simple_token,
                             extensions=extensions,
                             ignore_prefixes=ignore_prefixes,
                             multi_image=multi_image)

class MetaMorphTokenImporter(SimpleTokenImporter):

    TOKEN_P = Token('P', type_code='i', length='+', prefix='',
                    name=DIMENSION_NAME_POSITION)
    TOKEN_T = Token('T', type_code='i', length='+', prefix='',
                    name=DIMENSION_NAME_TIME)
    TOKEN_C = Token('C', type_code='c', length='+', prefix='',
                    name=DIMENSION_NAME_CHANNEL, regex_type='\D')
    TOKEN_Z = Token('Z', type_code='i', length='+', prefix='',
                    name=DIMENSION_NAME_ZSLICE)

    def __init__(self, path, separator='_',
                 extensions=None, ignore_prefixes=None, multi_image=None):
        super(MetaMorphTokenImporter,
              self).__init__(path, separator=separator,
                             extensions=extensions,
                             ignore_prefixes=ignore_prefixes,
                             multi_image=multi_image)

class ZeissLifeTokenImporter(SimpleTokenImporter):

    TOKEN_P = Token('s', type_code='i', length='+', prefix='',
                    name=DIMENSION_NAME_POSITION)
    TOKEN_T = Token('t', type_code='i', length='+', prefix='',
                    name=DIMENSION_NAME_TIME)
    TOKEN_C = Token('w', type_code='c', length='+', prefix='',
                    name=DIMENSION_NAME_CHANNEL)
    TOKEN_Z = Token('Z', type_code='i', length='+', prefix='',
                    name=DIMENSION_NAME_ZSLICE)

    def __init__(self, path, separator='_',
                 extensions=None, ignore_prefixes=None,
                 multi_image=None):
        super(ZeissLifeTokenImporter,
              self).__init__(path, separator=separator,
                             extensions=extensions,
                             ignore_prefixes=ignore_prefixes,
                             multi_image=multi_image)

class FlatFileImporter(FileTokenImporter):

    def __init__(self, path, filename,
                 extensions=None, ignore_prefixes=None,
                 multi_image=None):
        self.flat_filename = filename
        super(FlatFileImporter,
              self).__init__(path, token_handler=None,
                             extensions=extensions,
                             ignore_prefixes=ignore_prefixes,
                             multi_image=multi_image)

    def _build_dimension_lookup(self, filename=None):
        if filename is None:
            filename = self.flat_filename

        df = DefaultCoordinates()

        lookup = {}

        file = open(filename, 'r')
        headers = None
        has_xy = False

        positions = []
        times = []
        channels = []
        zslices = []

        for line in file.xreadlines():

            if headers is None:
                headers = [x.strip() for x in line.split('\t')]
                continue

            image_coord = dict(zip(headers, [x.strip() for x in line.split('\t')]))
            if not has_xy:
                has_xy = True
                info = ccore.ImageImportInfo(os.path.join(self.path,
                                                    image_coord['path'],
                                                    image_coord['filename']))
                self.meta_data.dim_x = info.width
                self.meta_data.dim_y = info.height
                self.meta_data.pixel_type = info.pixel_type
                self.has_multi_images = info.images > 1

            position = image_coord[DIMENSION_NAME_POSITION]
            if not position in lookup:
                lookup[position] = {}
            time = df(image_coord, DIMENSION_NAME_TIME)
            #time = image_coord[DIMENSION_NAME_TIME]
            if not time in lookup[position]:
                lookup[position][time] = {}
            channel = df(image_coord, DIMENSION_NAME_CHANNEL)
            if not channel in lookup[position][time]:
                lookup[position][time][channel] = {}
            zslice = df(image_coord, DIMENSION_NAME_ZSLICE)
            if not zslice in lookup[position][time][channel]:
                lookup[position][time][channel][zslice] = os.path.join(image_coord['path'], image_coord['filename'])

            timestamp = os.path.getmtime(os.path.join(self.path, os.path.join(image_coord['path'], image_coord['filename'])))
            self.meta_data.append_absolute_time(position,
                                                time,
                                                timestamp)

            if image_coord.has_key(META_INFO_WELL) and image_coord.has_key(META_INFO_SUBWELL):
                well = image_coord[META_INFO_WELL]
                subwell = image_coord[META_INFO_SUBWELL]

            well = df(image_coord, META_INFO_WELL)
            subwell = df(image_coord, META_INFO_SUBWELL)
            self.meta_data.append_well_subwell_info(position, well, subwell)

            if (self.has_multi_images and
                self.multi_image == self.MULTIIMAGE_USE_ZSLICE):
                if not zslice is None:
                    raise ValueError('Multi-image assigned for zslice conflicts'
                                     ' with zslice token in filename!')
                zslices.extend(range(1,info.images+1))
            else:
                zslices.append(zslice)

            positions.append(position)
            times.append(time)
            channels.append(channel)
        file.close()

        self.meta_data.positions = tuple(sorted(unique(positions)))
        self.meta_data.times = tuple(sorted(unique(times)))
        self.meta_data.channels = tuple(sorted(unique(channels)))
        self.meta_data.zslices = tuple(sorted(unique(zslices)))

        return lookup


class LsmImporter(object):

    def __init__(self):
        pass
