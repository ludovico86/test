

import numpy as np
from concert.quantities import q
from concert.base import Parameter
from concert.devices.cameras.base import Camera
from concert.devices.motors.base import Motor
from imwin import ImVis


def center_of_mass(frame):
    """Calculates the center of mass"""

    frm_shape = np.array(frame.shape)
    total = frame.sum()
    if total == 0:
        return np.array([-1,-1])
    else:
        y = (frame.sum(1)*np.arange(frm_shape[0])).sum() / total
        x = (frame.sum(0)*np.arange(frm_shape[1])).sum() / total
        return np.array([y,x])


def converge(do_list, drt, max_iter, init=None, final=None, result=[0]):
    """Repeat all functions in do_list until function drt (destination reached
    test) returns True (converge returns True) or max_iter is reached (converge
    returns False).
    
    Functions in do_list needs to take exactly one argmuent (that can be None).
    The Result of one function-call is passed through this argument to the next
    function-call.
    Funtion drt decides (using this results or not) if the destination is
    reached (converged = True) or not, but this result is not passed on, so
    it is possible to keep information about a state through the
    function-calls. The result of the last function-call of drt before
    returning is returned by converge.
    
    Optional argument init (initializition) must be callable which is called
    (without arguments) before starting the iteration over the do_list and
    final (finalization) is called before returning with the drts last result
    as first argument and do_lists last result as second argument.
    """
    
    tmp = result
    try
        tmp[0] = init()
    except TypeError:
        tmp[0] = None
    
    c = 0
    while c < max_iter:
        c += 1
        for x in do_list:
            tmp[0] = x(tmp[0])
        res = drt(tmp[0])
        if res:
            break
    
    try:
        tmp[0] = final(res, tmp[0])
    except TypeError:
        pass
    
    return res


def gauss_spot(shape, sigma, coord=[0, 0], rel_sec=1., m_axis=[1, 0]):
    """Creates an ndarray (image, dtype=float64, max()=1) in form of
    shape with a gaussian spot.

    If sigma is a scalar and rel_sec==1 (m_axis ignored), the spot will
    be a circle and coord can be used to set the position of the spot.
    rel_sec!=1 defines the relative scale-factor of the second axis to sigma.
    m_axis defines the axis of sigma (second axis is orthogonal to m_axis).

    If sigma is a 2x2-array-like, the shape of the spot is defined by
    sigma and coord can be used to position the spot (rel_sec and m_axis
    ignored).

    If sigma is a 3x3-array-like, all properties of the spot are defined
    by sigma.
    """

    def circle():
        x = np.arange(shape[0], dtype=np.float64) - coord[0]
        y = np.arange(shape[1], dtype=np.float64) - coord[1]
        return ((x**2)/float(sigma**2)).reshape(shape[0], 1) + \
            (y**2)/float(sigma**2)

    def ellipse2():
        vecs = np.indices(shape)
        vecs.shape = (2, 1, shape[0]*shape[1])
        vecs[0, 0, ...] -= coord[0]
        vecs[1, 0, ...] -= coord[1]
        vecs = vecs.repeat(2, 1)
        sh = sigma.shape
        b = [sigma.flat[x] for x in range(sigma.size)]
        b = b * (shape[0]*shape[1])
        s = np.array(b)
        s.shape = (shape[0]*shape[1], sh[1], sh[0])
        s = s.swapaxes(0, 2)
        sv = s * vecs
        vsv = vecs.swapaxes(0, 1) * sv
        return np.sum(np.sum(vsv, 0), 0).reshape(shape[0], shape[1])

    def ellipse3():
        vecs = np.append(np.indices(shape), np.ones(shape[0]*shape[1]))
        vecs.shape = (3, 1, shape[0]*shape[1])
        vecs = vecs.repeat(3, 1)
        sh = sigma.shape
        b = [sigma.flat[x] for x in range(sigma.size)]
        b = b * (shape[0]*shape[1])
        s = np.array(b)
        s.shape = (shape[0]*shape[1], sh[1], sh[0])
        s = s.swapaxes(0, 2)
        sv = s * vecs
        vsv = vecs.swapaxes(0, 1) * sv
        return np.sum(np.sum(vsv, 0), 0).reshape(shape[0], shape[1])

    from numbers import Number
    if isinstance(sigma, Number):
        if rel_sec == 1:
            return np.exp(-0.5*circle())
        ax = np.array(m_axis, np.float64)
        ax /= np.sqrt((ax**2).sum())
        v = np.mat([ax, [-ax[1], ax[0]]])
        sigma *= sigma
        d = np.eye(2) * np.array([sigma, sigma*rel_sec])
        sigma = np.mat(v.T * d * v)
    else:
        sigma = np.mat(sigma)

    if sigma.shape == (2, 2):
        d = np.linalg.det(sigma)
        if d <= 0:
            print 'gauss_spot: sigma defines not an ellipse'
        sigma = np.linalg.inv(sigma)
        return np.exp(-0.5*ellipse2())

    if sigma.shape == (3, 3):
        d = np.linalg.det(sigma)
        if d <= 0:
            print 'gauss_spot: sigma defines not an ellipse'
        sigma = np.linalg.inv(sigma)
        return np.exp(-0.5*ellipse3())

    raise ValueError('sigma can only be a scalar, 2x2 or 3x3-array-like')


def noise(ndArr, add_noise=(0, 0), mult_noise=(1, 1), snp_noise=(0, 0, 0),
          ins=False):
    """
    Adds noise to the Data saved in ndArr.

    Info: add_noise is applied after mult_noise and before snp_noise
    add_noise  = Array-like (min_val, max_val)
                 Creates Noise by adding a random-value from the range
                 min_val to max_val to the local data-value. min_val and
                 max_val may be 0, 1 or 2 dimensional (as long as
                 broadcasting-rules of numpy applicable).

    mult_noise = Array-like (min_fact, max_fact)
                 Creates Noise by multiplying a random-value between
                 min_fact and max_fact with the local data-value. min_val and
                 max_val may be 0, 1 or 2 dimensional (as long as
                 broadcasting-rules of numpy applicable).

    snp_noise  = Array-like (percentage, minvalue, maxvalue, balance) that
                 determines how much (percentage; between [0,1]) and
                 how (min-, maxvalue) data will be overridden for
                 "salt'n'pepper"-noise. percentage may be 0, 1 or 2
                 dimensional (as long as broadcasting-rules of numpy
                 applicable) with percentage.sum() determining the
                 percentage of how many pixels in total get manipulated and
                 the elementvalues determining the probability of a row/column
                 (1.dim.) or a pixel (2.dim) being manipulated. Optional
                 parameter balance adjusts the balance between min- and max-
                 value-manipulations (0: minvalue only; default 0.5: equal
                 number of min-/maxvalues; 1: maxvalue only).

    ins        = Boolean-Value
                 If true the function modifies ndArr, else a copy is modified.
    """

    if ins:
        a = ndArr
    else:
        a = ndArr.copy()

    mn = np.array(mult_noise)
    if (mn != 1).any():
        a *= np.random.random(a.shape) * (mn[1]-mn[0]) + mn[0]

    an = np.array(add_noise)
    if (an[0] != 0).any():
        a += np.random.random(a.shape) * (an[1]-an[0]) + an[0]

    n = np.array(snp_noise[0])
    if n.sum() > 0:
        if n.sum() > 1:
            n /= n.sum()
        bal = 0.5 if len(snp_noise) <= 3 else snp_noise[3]
        count = int(a.size*n.sum())
        val = np.array(snp_noise[1:3])
        snps = val[(np.random.random(count) < bal).astype(np.uint8)]
        rval = np.sort(np.append(np.random.random(count), 1))
        nc = np.append(0, np.broadcast_arrays(a, n)[1][:-2])
        nc = nc.copy().cumsum()
        nc /= nc[-1]

        ind = np.zeros(nc.size, np.bool_)
        i1 = 0
        i2 = 0
        while i1 < nc.size:
            #print i1, i2, nc[i1], '>', rval[i2],
            ind[i1] = nc[i1] > rval[i2]
            #print ind[i1]
            if ind[i1]:
                i2 += 1
            i1 += 1

        a.flat[ind] = snps

    return a


class DummyCamera(Camera):
    """DummyCamera that returns an image depending on its coordination-setting.
    Images of a gaussian-spot can be created (with a spot centered in the
    image at coord [0,0]) or a cutout of a image loaded from a file (areas
    outside that image are filled with zeros)(attention: coord [0,0] is center
    of frame).
    """

    def __init__(self, pos=(0., 0.), imgsize=(640, 480)):
        params = [Parameter('exposure-time', unit=q.s),
                    Parameter('roi-width'),
                    Parameter('roi-height'),
                    Parameter('sensor-pixel-width', unit=q.micrometer),
                    Parameter('sensor-pixel-height', unit=q.micrometer)]
        super(DummyCamera, self).__init__(params)
        self.exposure_time = 1 * q.ms
        self.sensor_pixel_width = 5 * q.micrometer
        self.sensor_pixel_height = 5 * q.micrometer
        self._sigma = 10
        self._rel_sec = 1.
        self._m_ax = [1, 0]
        self._imgshape = imgsize
        self._noise = None
        self._frm_mode = 'generate'

        self._coord = [pos[0] * q.meter, pos[1] * q.meter]
        self._coord_at_trigger = [pos[0] * q.meter, pos[1] * q.meter]

    def _record_real(self):
        pass

    def _stop_real(self):
        pass

    def _trigger_real(self):
        self._coord_at_trigger = self._coord

    def _grab_real(self):
        c = [(self._coord_at_trigger[0] / self.sensor_pixel_width).
            to(q.dimensionless).magnitude,
            (self._coord_at_trigger[1] / self.sensor_pixel_height).
            to(q.dimensionless).magnitude]
        #print 'DummyCam.grab(): coord', c
        if self._frm_mode == 'file':
            tp = self._image.dtype
            #print 'image.type =', tp
            src_offset = np.array(self._image.shape)/2
            dst_offset = np.array(self._imgshape)/2
            src_start = np.round(src_offset + c - dst_offset)
            src_stop = np.round(src_offset + c + dst_offset)
            if (src_start >= self._image.shape).any() or\
                    (src_stop < 0).any():
                img = np.zeros(self._imgshape)
            else:
                test1 = src_start < 0
                dst_start = np.where(test1, -src_start, 0)
                src_start[test1] = 0
                test2 = src_stop >= self._image.shape
                dst_stop = np.where(test2, src_offset*2-src_stop-1,
                                    self._imgshape)
                src_stop[test2] = np.array(self._image.shape)[test2]-1
                img = np.zeros(self._imgshape)
                
                img[dst_start[0]:dst_stop[0], dst_start[1]:dst_stop[1]] =\
                    self._image[src_start[0]:src_stop[0], 
                                src_start[1]:src_stop[1]]
        else:
            time = self.exposure_time.to(q.s).magnitude
            img = gauss_spot(self._imgshape, self._sigma, c, self._rel_sec,
                            self._m_ax) * time*1000*(2**16-1)
            tp = np.uint16

        if not self._noise is None:
            img = noise(img, **self._noise)
            try:
                mini, maxi = np.iinfo(tp).min, np.iinfo(tp).max  
                img[img < mini] = mini
                img[img > maxi] = maxi
            except ValueError:
                pass
        return np.require(img, tp)

    def from_image(self, img):
        self._image = img
        self._imgshape = img.shape
        self._frm_mode = 'file'
        self._coord = np.array([0,0])*q.micrometer

class DummyMotor(Motor):
    """A dummymotor that updates the coordates of a dummycamera on motor-
    movement."""

    def __init__(self, cam, axis, calibration=None, limiter=None,
                    position=0, hard_limits=None):
        super(DummyMotor, self).__init__(calibration, limiter)
        self._remoteCam = cam
        self._axis = axis
        self._position = position
        self._hard_limits = (-100, 100) if hard_limits is None else \
            hard_limits

    def _set_position(self, position):
        if position < self._hard_limits[0]:
            self._position = self._hard_limits[0]
        elif not position < self._hard_limits[1]:
            # https://github.com/hgrecco/pint/issues/40
            self._position = self._hard_limits[1]
        else:
            diff = position - self._position
            self._position = position
            d1 = self._axis[0]*diff
            d2 = self._axis[1]*diff
            self._remoteCam._coord[0] += d1*q.mm
            self._remoteCam._coord[1] += d2*q.mm

    def _get_position(self):
        return self._position

    def _stop_real(self):
        pass

    def _home(self):
        self._set_position(0)

    def in_hard_limit(self):
        return self._position < self._hard_limits[0] or not \
            self._position < self._hard_limits[1]


def diff2center_of_mass(img):
    """"""

    return center_of_mass(img-img.min()) + [0.5, 0.5] - img.shape


def converge_test(move_methode, cam_data_ff='stitched_image.jpg', tolerance=5,
                  max_iter=100, imgnoise=None, save_to='converge_test.npz'):
    """"""

    def cam2pixel():
        r = np.array(cam._coord)
        r /= [cam.sensor_pixel_height, cam.sensor_pixel_width]
        r += [0.5, 0.5] - np.array(frm_shape)/2
        return r

    def pixel2cam(pix):
        cam._coord = [pix[0]*cam.sensor_pixel_height,
                      pix[1]*cam.sensor_pixel_width]

    def initializition():
        xmot.position = 0*q.m
        zmot.position = 0*q.m
        pixel2cam([yrang[yind],x])
        return 0

    def do(arg1):
        cam.trigger()
        img = cam.grab()
        move = move_methode(img)
        xmot.move(move[0]*cam.sensor_pixel_height)
        zmot.move(move[1]*cam.sensor_pixel_width)
        return arg1 + 1

    def dest_reached(arg1):
        coord = cam2pixel()
        return ((coord - beamcenter)**2).sum() < tol2

    # testparam init
    beamcenter = center_of_mass(img-img.min())
    tol2 = tolerance ** 2

    # init
    from scipy.ndimage import imread
    cam = DummyCamera()
    if isinstance(cam_data_ff, str) and len(cam_data_ff) > 0:
        img = imread(cam_data_ff)
    else:
        cam_data_ff = 'generated'
        o_sh = cam._imgshape
        cam._imgshape = (6000,8000)
        cam._sigma = 1000
        cam._rel_sec = 0.4
        cam._m_ax = [1, 0]
        cam.trigger()
        img = cam.grab()
        cam._imgshape = o_sh
    frm_shape = img.shape
    cam.from_image(img)
    cam._noise = imgnoise
    xmot = DummyMotor(cam, [1, 0])
    zmot = DummyMotor(cam, [0, 1])

    # progressbar init
    prog = np.zeros((50,400), np.uint8)
    ImVis.swapaxis = False
    title_txt = 'Running test: '
    percent = 0
    progbar = ImVis({'ndArr': prog, 'imgTitle': title_txt+str(percent)+'%'})

    # result init
    to_gray = lambda x: (x*256 + x)*256 +x
    gray = to_gray(127)
    white = to_gray(255)
    red = 255 * 2**16
    green = 255 * 2**8
    img_dr = np.tile(gray, frm_shape, dtype=np.uint32)  # destination reached
    img_rc = np.tile(red, frm_shape, dtype=np.uint32)   # reachcount
    img_ep = np.zeros(frm_shape, np.uint32)             # endpoint

    # testloop
    ylen = frm_shape[0]-cam._imgshape[0]
    ystart = round(cam._imgshape[0]/2)
    yrang = np.arange(ystart, ystart-ylen)
    xlen = frm_shape[1]-cam._imgshape[1]
    xstart = round(cam._imgshape[1]/2)
    xrang = np.arange(xstart,xstart+xlen)
    _iter = [0]
    for yind in range(ylen):
        rel = yind/ylen
        prog[:, :400*rel] = 255
        progbar.image(prog)
        progbar.setTitle(title_txt+str(round(100*rel, 1))+'%')
        for x in xrang:
            res = converge([do], dest_reached, max_iter, initializition,
                           None, _iter)
            img_dr[yrang[yind],x] = green if res else red
            img_rc[yrang[yind],x] = to_gray(int(_iter[0]*255./max_iter))
            endpoint = cam2pixel()
            if not ((endpoint < 0).any() or (endpoint > frm_shape).any()):
                img_ep[endpoint[0], endpoint[1]] = green if res else white

    # save to file
    np.savez(save_to,
             main_img=np.require(img, np.dtype('<i4')),
             img_dr=np.require(img_dr, np.dtype('<i4')),
             img_rc=np.require(img_rc, np.dtype('<i4')),
             img_ep=np.require(img_ep, np.dtype('<i4')))#,
             #ext_data=#TODO)

    # visualize
    i1 = {'ndArr': img, 'imgTitle': 'Cam_Data: ', 'imgInfo': 'Main image ('+
          cam_data_ff+', '+img.shape[1]+'x'+img.shape[0]+'pixels) that is '+
          'used as a template.'+
          '\nThe viewport of the cutout image is '+str(cam._imgshape[1])+'x'+
          str(cam._imgshape[0])+'pixels'+
          '\nnoise = '+str(imgnoise)+
          '\ntolerance = '+str(tolerance)+'pixels'+
          '\nmax_iterations = '+str(max_iter)}
    i2 = {}


def converge_test_ff(filename='converge_test.npz'):
    """"""

    


























