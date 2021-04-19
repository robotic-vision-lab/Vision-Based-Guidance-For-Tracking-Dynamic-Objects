import os
import sys



# from settings import *                                      #pylint: disable=unused-wildcard-import
from .optical_flow_config import (FARNEBACK_PARAMS,          #pylint: disable=unused-import
                                 FARN_TEMP_FOLDER,
                                 FEATURE_PARAMS,
                                 LK_PARAMS,
                                 LK_TEMP_FOLDER,
                                 MAX_NUM_CORNERS)

from .game_utils import (load_image,                         #pylint: disable=unused-import
                        _prep_temp_folder,
                        vec_str,
                        scale_img,
                        ImageDumper)
# add vca\ to sys.path
vca_path = os.path.abspath(os.path.join('..'))
if vca_path not in sys.path:
    sys.path.append(vca_path)

from utils.vid_utils import create_video_from_images
from utils.optical_flow_utils \
                import (get_OF_color_encoded,               #pylint: disable=unused-import
                        draw_sparse_optical_flow_arrows,
                        draw_tracks)
from utils.img_utils import (convert_to_grayscale,          #pylint: disable=unused-import
                             convert_grayscale_to_BGR,
                             put_text,
                             draw_point,
                             images_assemble,
                             add_salt_pepper)
from utils.img_utils import scale_image as cv_scale_img

from algorithms.optical_flow \
                import (compute_optical_flow_farneback,     #pylint: disable=unused-import
                        compute_optical_flow_HS,
                        compute_optical_flow_LK)

from algorithms.feature_detection \
                import (Sift,)

from algorithms.feature_match \
                import (BruteL2,)

from algorithms.template_match \
                import (CorrelationCoeffNormed,
                        TemplateMatcher)