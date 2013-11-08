"""
classfinder.py

Classes for image segmenentation, feature extraction, and classification.
If object of interset is found send to WindowsRegistry imaging job Trigger1.
Output is written in a common directory
This class should be replaced by multijob.py
"""
import traceback
import os
import sys
import numpy as np
from collections import OrderedDict
from pylsm.lsmreader import Lsmimage

# to find the settingsmapper
sys.path.append(os.path.dirname(__file__))

from os.path import basename, splitext, join

try:
    import cecog
except ImportError:
    sys.path.append("../../pysrc")
    import cecog

from cecog import ccore
from cecog import CH_VIRTUAL
from cecog.traits.analyzer import SECTION_REGISTRY
from cecog.learning.learning import CommonClassPredictor
from cecog.environment import CecogEnvironment
from cecog.io.imagecontainer import MetaImage
from cecog.analyzer.channel import PrimaryChannel
from cecog.analyzer.channel import SecondaryChannel
from cecog.analyzer.channel import TertiaryChannel
from cecog.analyzer.channel import MergedChannel
from cecog.traits.config import ConfigSettings

# hexToRgb was replace in cellcogniton 1.5.0 since
# matplotlib implements the same functionality
try:
    from cecog.util.util import hexToRgb
except ImportError:
    from cecog.colors import hex2rgb as hexToRgb


class SettingsMapper(object):
    """Map parameters from a ConfigSettings instance to groups to fit
    the API"""

    CHANNEL_CLASSES = (PrimaryChannel, SecondaryChannel,
                        TertiaryChannel, MergedChannel)

    FEATURES = {'featurecategory_intensity': ['normbase', 'normbase2'],
                'featurecategory_haralick': ['haralick', 'haralick2'],
                'featurecategory_stat_geom': ['levelset'],
                'featurecategory_granugrey': ['granulometry'],
                'featurecategory_basicshape': ['roisize',
                                               'circularity',
                                               'irregularity',
                                               'irregularity2',
                                               'axes'],
                'featurecategory_convhull': ['convexhull'],
                'featurecategory_distance': ['distance'],
                'featurecategory_moments': ['moments']}

    def __init__(self, configfile):
        self.img_height = None
        self.img_width = None
        self.settings = ConfigSettings(SECTION_REGISTRY)
        self.settings.read(configfile)

    def __call__(self, section, param):
        return self.settings.get(section, param)

    def setImageSize(self, width, height):
        self.img_width = width
        self.img_height = height

    @property
    def img_size(self):
        return self.img_width, self.img_height

    def featureParams(self, ch_name="Primary"):
        f_categories = list()
        f_cat_params = dict()

        # unfortunateley some classes expect empty list and dict
        if ch_name.lower() in CH_VIRTUAL:
            return f_categories, f_cat_params

        for cat, feature in self.FEATURES.iteritems():
            featopt = '%s_%s' %(ch_name, cat)
            if self('FeatureExtraction', featopt):
                if "haralick" in cat:
                    try:
                        f_cat_params['haralick_categories'].extend(feature)
                    except KeyError:
                        assert isinstance(feature, list)
                        f_cat_params['haralick_categories'] = feature
                else:
                    f_categories += feature

        if f_cat_params.has_key("haralick_categories"):
            f_cat_params['haralick_distances'] = (1, 2, 4, 8)

        return f_categories, f_cat_params

    def zsliceParams(self, chname):
        self.settings.set_section('ObjectDetection')
        if self("ObjectDetection", "%s_%s" %(chname.lower(), 'zslice_selection')):
            par = self("ObjectDetection", "%s_%s" %(chname.lower(), 'zslice_selection_slice'))
        elif self("ObjectDetection", "%s_%s" %(chname.lower(), 'zslice_projection')):
            method = self("ObjectDetection", "%s_%s" %(chname.lower(), 'zslice_projection_method'))
            begin = self("ObjectDetection", "%s_%s" %(chname.lower(), 'zslice_projection_begin'))
            end = self("ObjectDetection", "%s_%s" %(chname.lower(), 'zslice_projection_end'))
            step = self("ObjectDetection", "%s_%s" %(chname.lower(), 'zslice_projection_step'))
            par = (method, begin, end, step)
        return par

    def registrationShift(self):
        xs = [0]
        ys = [0]

        for prefix in (SecondaryChannel.PREFIX, TertiaryChannel.PREFIX):
            if self('Processing','%s_processchannel' %prefix):
                reg_x = self('ObjectDetection', '%s_channelregistration_x' %prefix)
                reg_y = self('ObjectDetection', '%s_channelregistration_y' %prefix)
                xs.append(reg_x)
                ys.append(reg_y)

        diff_x = []
        diff_y = []
        for i in range(len(xs)):
            for j in range(i, len(xs)):
                diff_x.append(abs(xs[i]-xs[j]))
                diff_y.append(abs(ys[i]-ys[j]))

        if self('General', 'crop_image'):
            y0 = self('General', 'crop_image_y0')
            y1 = self('General', 'crop_image_y1')
            x0 = self('General', 'crop_image_x0')
            x1 = self('General', 'crop_image_x1')

            self.img_width = x1 - x0
            self.img_height = y1 - y0

        if self.img_height is None or self.img_width is None:
            raise RuntimeError("Images size is not set. Use self.setImageSize(*size)")

        # new image size after registration of all images
        image_size = (self.img_width - max(diff_x),
                      self.img_width - max(diff_y))

        return (max(xs), max(ys)), image_size


    def channelParams(self, chname="Primary", color=None):
        f_cats, f_params = self.featureParams(chname)
        shift, size = self.registrationShift()
        params = {'strChannelId': color,
                  'channelRegistration': (self(
                    'ObjectDetection', '%s_channelregistration_x' %chname),
                                          self(
                    'ObjectDetection', '%s_channelregistration_y' %chname)),
                  'oZSliceOrProjection': self.zsliceParams(chname),
                  'new_image_size': size,
                  'registration_start': shift,
                  'fNormalizeMin': self('ObjectDetection', '%s_normalizemin' %chname),
                  'fNormalizeMax': self('ObjectDetection', '%s_normalizemax' %chname),
                  'lstFeatureCategories': f_cats,
                  'dctFeatureParameters': f_params}
        return params

    def channelRegions(self):
        """Return a dict of channel region pairs according to the classifier."""

        regions = OrderedDict()
        for ch_cls in self.CHANNEL_CLASSES:
            name = ch_cls.NAME
            if not ch_cls.is_virtual():
                region = self( \
                    "Classification", "%s_classification_regionname" %(name))

                # no plugins loaded
                if region not in (None, ""):
                    regions[name] = region
            else:
                regions2 = OrderedDict()
                for ch_cls2 in self.CHANNEL_CLASSES:
                    if ch_cls2.is_virtual():
                        continue
                    name2 =  ch_cls2.NAME
                    if self("Classification", "merge_%s" %name2):
                        regions2[name2] = self("Classification", "%s_%s_region" %(name, name2))
                if regions2:
                    regions[name] = regions2
        return regions


class LsmImage(Lsmimage):
    """LSM image class to fit the needs of the classfinder plugin.
    i.e. it has methods to return the number of channels, z-slices etc...
    """

    CZ_LSM_INFO = 'CZ LSM info'
    WIDTH = 'Dimension X'
    HEIGHT = 'Dimension Y'
    ZSTACK = 'Dimension Z'
    CHANNEL = 'Sample / Pixel'
    IMAGE = 'Image'

    def __init__(self, *args, **kw):
        Lsmimage.__init__(self, *args, **kw)

    @property
    def zslices(self):
        return self.header[self.CZ_LSM_INFO][self.ZSTACK]

    @property
    def size(self):
        return (self.header[self.CZ_LSM_INFO][self.WIDTH],
                self.header[self.CZ_LSM_INFO][self.HEIGHT])

    @property
    def channels(self):
        return self.header[self.IMAGE][0][self.CHANNEL]

    def meta_images(self, channel):
        """Get a list of cellcognition meta images for a given channel.
        One meta image per z-slice"""
        assert isinstance(channel, int)

        if not (0 <= channel < self.channels):
            raise RuntimeError("channel %d does not exist" %channel)

        metaimages = list()
        for i in xrange(self.zslices):
            img = self.get_image(stack=i, channel=channel)
            # kinda sucks, but theres no way around
            metaimage = MetaImage()
            metaimage.set_image(ccore.numpy_to_image(img, copy=True))
            metaimages.append(metaimage)
        return metaimages


class ImageProcessor(object):

    def __init__(self, mapper, imagefile):
        super(ImageProcessor, self).__init__()
        self.mapper = mapper
        self._channels = OrderedDict()

        self.image = LsmImage(imagefile)
        self.image.open()
        self.mapper.setImageSize(*self.image.size)
        self._setupChannels()

    def _setupChannels(self):
        chdict = dict((c.NAME.lower(), c) for c in self.mapper.CHANNEL_CLASSES)

        for cname, region in self.mapper.channelRegions().iteritems():
            # use a mapping if one exists
            cid = self.mapper("ObjectDetection", "%s_channelid" %cname)
            channel = chdict[cname.lower()](
                **self.mapper.channelParams(cname.title(), cid))

            if channel.is_virtual():
                channel.merge_regions = region
            else:
                channel.SEGMENTATION.init_from_settings(self.mapper.settings)
                for zslice in self.image.meta_images(eval(cid)-1):
                    channel.append_zslice(zslice)

            self._channels[cname] = channel

    def exportLabelImage(self, ofile, cname):
        channel = self._channels[cname]
        for region in channel.region_names():
            container = channel.containers[region]
            if isinstance(region, tuple):
                region = '-'.join(region)
            lif_name = ofile+"-lables_%s_%s.tif" %(cname.lower(), region)
            print(lif_name)
            container.exportLabelImage(lif_name, "LWZ")

    def exportClassificationImage(self, ofile, cname):
        channel = self._channels[cname]
        for region in channel.region_names():
            container = channel.containers[region]
            if isinstance(region, tuple):
                region = '-'.join(region)
            ofile = ofile+"-classification_%s_%s.tif" %(cname.lower(), region)
            print('saving %s' %ofile)
            container.exportRGB(ofile, '90')

    def process(self):
        """process files: create projection normalize image get objects for
        several channels"""
        channels = list()
        for cname, channel in self._channels.iteritems():
            channels.append(channel)

            channel.apply_zselection()
            channel.normalize_image()
            channel.apply_registration()

            if isinstance(channel, PrimaryChannel):
                channel.apply_segmentation()
            elif isinstance(channel, (SecondaryChannel, TertiaryChannel)):
                channel.apply_segmentation(*channels[:])
            elif isinstance(channel, MergedChannel):
                channel.apply_segmentation(self._channels,
                                           master=PrimaryChannel.NAME)
            channel.apply_features()

    def findObjects(self, classifier):
        """ runs the classifier """
        objects = list()
        probs = list()
        channel = self._channels[classifier.name.title()]

        for region in channel.region_names():
            holder = channel.get_region(region)
            container = channel.containers[region]
            for l, obj in holder.iteritems():

                obj.iLabel, prob = classifier.predict(obj.aFeatures,
                                                   holder.feature_names)

                obj.strClassNames = classifier.class_names[obj.iLabel]
                objects.append(obj)
                probs.append(prob)

                # for classification images
                hexcolor = classifier.hexcolors[ \
                    classifier.class_names[obj.iLabel]]
                rgb = ccore.RGBValue(*hexToRgb(hexcolor))
                container.markObjects([l], rgb, False, True)

        return np.array(objects), np.array(probs)


class ClassFinder(object):

    def __init__(self, imagefile, class_of_interest, classifier_name, cfgfile,
                 outdir=os.getcwd(), out_images=True):

        super(ClassFinder, self).__init__()
        self.environ = CecogEnvironment(cecog.VERSION,
                                        redirect=False, debug=False)
        self.mapper = SettingsMapper(cfgfile)
        self.class_of_interest = class_of_interest
        self.out_images = out_images
        self.classifier_name = classifier_name
        self.outdir = outdir
        self.imagefile = imagefile
        self._setupClassifier()


    def _setupClassifier(self):
        self.classifier = CommonClassPredictor( \
            clf_dir=self.mapper('Classification',
                                '%s_classification_envpath'
                                %self.classifier_name),
            name=self.classifier_name,
            channels=self.classifier_name,
            color_channel=self.mapper("ObjectDetection", "%s_channelid"
                                      %self.classifier_name))
        self.classifier.importFromArff()
        self.classifier.loadClassifier()

        if self.class_of_interest not \
                in self.classifier.class_names.keys():
            raise RuntimeError("Class of interest is not defined!")


    def __call__(self):
        ofile = join(self.outdir, str(splitext(basename(self.imagefile))[0]))
        imp = ImageProcessor(self.mapper, str(self.imagefile))
        imp.process()

        print('processing image %s' %basename(self.imagefile))
        candidates, probs = imp.findObjects(self.classifier)

        # after classification!
        if self.out_images:
            imp.exportLabelImage(ofile, self.classifier_name.title())
            imp.exportClassificationImage(ofile, self.classifier_name.title())

        classname = self.classifier.class_names[self.class_of_interest]
        import pdb; pdb.set_trace()


if __name__ == '__main__':

    params = {'imagefile': '/Users/hoefler/sandbox/pycronaut/test_data/test_image_merged.lsm',
              'cfgfile': '/Users/hoefler/sandbox/pycronaut/test_data/test_merged_channel.conf',
              'class_of_interest': 4,
              'out_images': False,
              'classifier_name': "Merged"}

    cf = ClassFinder(**params)
    cf()
