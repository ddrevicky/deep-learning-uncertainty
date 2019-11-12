from pathlib import Path
import numpy as np
from scipy import ndimage

# Cephallometry dataset properties
ORIG_IMAGE_X = 1935
ORIG_IMAGE_Y = 2400
PIXELS_PER_MM = 10
N_LANDMARKS = 19

def list_files(dir_path):
    return sorted(list(dir_path.iterdir()))

def get_elastic_transform_coordinates(im, sigma=8.0, alpha=15.0, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    dx = random_state.rand(*im.shape) * 2 - 1
    dy = random_state.rand(*im.shape) * 2 - 1
    dx = ndimage.gaussian_filter(dx, sigma, mode='constant', cval=0) * alpha
    dy = ndimage.gaussian_filter(dy, sigma, mode='constant', cval=0) * alpha
    x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    coordinates = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    return coordinates, dx, dy

class ElasticTransform():
    def __init__(self, sigma=8.0, alpha=15.0):
        self.sigma = sigma
        self.alpha = alpha
        
    def get_coordinates(self, im):
        dx = np.random.rand(*im.shape) * 2 - 1
        dy = np.random.rand(*im.shape) * 2 - 1
        dx = ndimage.gaussian_filter(dx, self.sigma, mode='constant', cval=0) * self.alpha
        dy = ndimage.gaussian_filter(dy, self.sigma, mode='constant', cval=0) * self.alpha
        x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))
        # For each coordinate in the new image we will look to the original image at x_coords, y_coords
        # and interpolate the new value from there
        x_coords, y_coords = x+dx, y+dy  
        return x_coords, y_coords, dx, dy

class AffineTransform():
    def __init__(self, angle, scales=None, tx=None, ty=None):
        self.scales = scales
        self.angle = angle
        self.tx = tx
        self.ty = ty
        
        translations = [tx, ty]
        for t in translations:
            if t is not None and not (0.0 <= t <= 1.0):
                raise ValueError("translation values should be between 0 and 1")
    
    @staticmethod
    def get_params(angle, scales, tx, ty, shape):
        height, width = shape
        
        angle = np.random.uniform(-angle, angle)
        scale = 1.0
        tx, ty = 0, 0
        if scales is not None:
            scale = np.random.uniform(scales[0], scales[1])
        if tx is not None:
            max_tx = tx*width
            tx = np.random.uniform(-max_tx, max_tx)
        if ty is not None:
            max_ty = ty*height
            ty = np.random.uniform(-max_ty, max_ty)

        return angle, scale, tx, ty
    
    @staticmethod
    def get_translation_matrix(tx, ty):
        m = np.eye(3)
        m[0, 2] = ty
        m[1, 2] = tx
        return m
    
    @staticmethod
    def get_affine_matrix(angle, scale, tx, ty, shape):
        ''' Calculates forward affine transform matrix.
        '''
        height, width = shape
        # Shift for rotation around center
        tr_fw = AffineTransform.get_translation_matrix(width/2, height/2)
        tr_bk = AffineTransform.get_translation_matrix(-width/2, -height/2)

        rot_scale = np.eye(3)
        angle = angle * (np.pi/180)
        cos = np.cos(angle)
        sin = np.sin(angle)
        rot_scale[0, 0] = scale * cos
        rot_scale[1, 0] = scale * sin
        rot_scale[1, 1] = scale * cos
        rot_scale[0, 1] = -scale * sin

        transl = AffineTransform.get_translation_matrix(tx, ty)

        return np.linalg.multi_dot([tr_fw, rot_scale, tr_bk, transl])
    
    def get_matrix(self, array, heatmap=False):
        angle, scale, tx, ty = self.get_params(self.angle, self.scales, self.tx, self.ty, array.shape)
        m = self.get_affine_matrix(angle, scale, tx, ty, array.shape)
        return m