

import weakref
import numpy as np
from concert.quantities import q
from concert.base import Parameter
from concert.devices.cameras.base import Camera
from concert.devices.motors.base import Motor
from imwin import ImVis, _ext_Callback


def center_of_mass(frame):
    """Calculates the center of mass"""

    frm_shape = np.array(frame.shape)
    total = frame.sum()
    if total == 0:
        return np.array([-1, -1])
    else:
        y = (frame.sum(1)*np.arange(frm_shape[0])).sum() / total
        x = (frame.sum(0)*np.arange(frm_shape[1])).sum() / total
        return np.array([y, x])


def converge(do_list, drt, max_iter, at=None, init=None, final=None,
             result=[0]):
    """Repeat all functions in do_list until function drt (destination reached
    test) returns True (converge returns "success") or function at (optional;
    abort test) returns True (converge returns "abort") or max_iter is reached
    (converge returns "fail").

    Functions in do_list needs to take exactly one argmuent (that can be None).
    The Result of one function-call is passed through this argument to the next
    function-call.
    Funtion drt decides (using this results or not) if the destination is
    reached (converged = "success") or not, but this result is not passed on,
    so it is possible to keep information about a state through the
    function-calls.

    Optional function at (using previous result) decides if a condition is
    meet where aborting speeds up computation by not iterating until max_iter
    is reached. True = converge returns "abort", False => continue computation.

    Optional argument init (initializition) must be callable which is called
    (without arguments) before starting the iteration over the do_list and
    final (finalization) is called before returning with the drts last result
    as first argument and do_lists last result as second argument.

    Optional argument result will hold the last result from internal function-
    call before returning.
    """

    tmp = result
    if at is None:
        at = lambda x: False
    try:
        tmp[0] = init()
    except TypeError:
        tmp[0] = None

    c = 0
    while c < max_iter:
        if at(tmp[0]):
            res = 'abort'
            break
        for x in do_list:
            tmp[0] = x(tmp[0])
        if drt(tmp[0]):
            res = 'success'
            break
        else:
            res = 'fail'
        c += 1

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

    def __init__(self, pos=(0., 0.), imgsize=(480, 640)):
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
        self._trace = None

        self._coord = [pos[0] * q.meter, pos[1] * q.meter]
        self._coord_at_trigger = [pos[0] * q.meter, pos[1] * q.meter]
        #self._callbacks = {'start_recording': [],
        #                   'stop_recording': [],
        #                   'trigger': [],
        #                   'grab': []}

    def _record_real(self):
        pass

    def _stop_real(self):
        pass

    def _trigger_real(self):
        self._coord_at_trigger = self._coord
        if not self._trace is None:
            self._trace.append([self._coord[x]*1 for x in range(2)])

    def _grab_real(self):
        return self.img_from_coord(self._coord_at_trigger)

    def img_from_coord(self, coord):
        c = [(coord[0]/self.sensor_pixel_width).to(q.dimensionless.units).\
             magnitude,
             (coord[1]/self.sensor_pixel_height).to(q.dimensionless.units).\
             magnitude]
        #print 'DummyCam.grab(): coord', c
        if self._frm_mode == 'file':
            src = self._image
            srcs = src.shape
            tp = src.dtype
            so = np.array(srcs)/2
            do = np.array(self._imgshape)/2
            ss = np.require(np.round(so + c - do), np.uint32)
            se = ss + self._imgshape
            if (ss >= srcs).any() or (se < 0).any():
                img = np.zeros(self._imgshape)
            else:
                t1 = ss < 0
                ds = np.where(t1, -ss, 0)
                ss[t1] = 0
                t2 = se >= srcs
                de = np.where(t2, srcs-se+self._imgshape, self._imgshape)
                se[t2] = np.array(srcs)[t2]
                img = np.zeros(self._imgshape)
                img[ds[0]:de[0], ds[1]:de[1]] = src[ss[0]:se[0], ss[1]:se[1]]
        else:
            time = self.exposure_time.to(q.s.units).magnitude
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
        self._image = img.copy()
        self._imgshape = img.shape
        print 'load image into DummyCamera:', self._imgshape
        self._frm_mode = 'file'
        self._coord = np.array([0., 0.])*q.um

    def start_tracing(self):
        tr = self._trace
        self._trace = []
        return tr
        
    def stop_tracing(self):
        tr = self._trace
        self._trace = None
        return tr


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

    def move(self, delta):
        ddelta = delta.to(q.mm.units).magnitude
        if self._position+ddelta < self._hard_limits[0]:
            ddelta = self._hard_limits[0] - self._position
        elif not self._position+ddelta < self._hard_limits[1]:
            # https://github.com/hgrecco/pint/issues/40
            ddelta = self._hard_limits[1] - self._position
        self._position += ddelta

        d1 = self._axis[0]*ddelta
        d2 = self._axis[1]*ddelta
        self._remoteCam._coord[0] += d1*q.mm
        self._remoteCam._coord[1] += d2*q.mm

    def _set_position(self, position):
        if position < self._hard_limits[0]:
            position = self._hard_limits[0]
        elif not position < self._hard_limits[1]:
            # https://github.com/hgrecco/pint/issues/40
            position = self._hard_limits[1]
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


#def 


def fcenter(img):
    """"""

    from scipy.ndimage.filters import gaussian_filter
    img = gaussian_filter(img-img.min(), 20)
    mask = img > (0.8*img.max())
    return center_of_mass(mask) + [0.5, 0.5] - np.array(img.shape)/2


def filtered_center(img):
    """"""

    from scipy.ndimage.filters import gaussian_filter
    img = gaussian_filter(img-img.min(), 20)
    return center_of_mass(img) + [0.5, 0.5] - np.array(img.shape)/2


def diff2center_of_mass(img):
    """"""

    return center_of_mass(img-img.min()) + [0.5, 0.5] - np.array(img.shape)/2


def ccolor(typ, val, txt):
    return {'type': typ, 'val': val, 'txt': txt}


colors = {'bc_inraw': ccolor('fix', 255*256, 'green'),
          'bc_indata': ccolor('fix', (127*256+127)*256+127, 'gray'),
          'bg': ccolor('fix', (64*256+64)*256+64, 'darkgray'),
          'u_sp': ccolor('fix', 0, 'black'),
          'dr_s': ccolor('gradient', lambda x: 255*2**8+x,
                         ['green to cyan with green = ', ' to cyan = ',
                          ' iterations']),
          'dr_f': ccolor('fix', (200*2**8+200)*2**8, 'yellow'),
          'dr_a': ccolor('gradient', lambda x: 255*2**16+x,
                         ['red to magenta with red = after ',
                          ' to magenta = after ', ' iterations aborted']),
          'ep_s': ccolor('gradient', lambda x: 255*2**8+x,
                         ['green to cyan with green = ', ' to cyan = ',
                          ' endpoints at the same position']),
          'ep_f': ccolor('fix', (200*2**8+200)*2**8, 'yellow'),
          'ep_a': ccolor('gradient', lambda x: 255*2**16+x,
                         ['red to magenta with red = ', ' to magenta = ',
                          ' failed endpoints at the same position'])}


def lcolor(colkey, x=0):
    return colors[colkey]['val'] if colors[colkey]['type'] == 'fix' else \
        colors[colkey]['val'](x)


def lcol_txt(colkey, start=0, end=0):
    return colors[colkey]['txt'] if colors[colkey]['type'] == 'fix' else \
        colors[colkey]['txt'][0]+str(start)+colors[colkey]['txt'][1] + \
        str(end)+colors[colkey]['txt'][2]


def converge_test(move_methode, cam_data='stitched_image.jpg',
                  tolerance=50, max_iter=100, imgnoise=None, max_time=86400,
                  trace=False):
    """"""
    import time

    def cam2pixel(coord=None):
        coord = cam._coord if coord is None else coord
        r = np.array(coord)
        r /= [cam.sensor_pixel_height, cam.sensor_pixel_width]
        r = np.array([r[0].magnitude, r[1].magnitude])
        r += frm_shape/2 - [0.5, 0.5]
        return r

    def pixel2cam(pix):
        pix -= frm_shape/2
        cam._coord = [pix[0]*cam.sensor_pixel_height,
                      pix[1]*cam.sensor_pixel_width]

    def print_progress():
        print str(round(tdiff, 1))+'s',
        if trel == 0:
            print ',\t',
        else:
            total = tdiff/trel
            print '(estm. full_test:'+str(round(total, 1))+'s, to go:' + \
                (str(round(total-tdiff, 1))+'s' if total < max_time else
                 str(round(max_time+add2timeout[0]-tdiff, 1)) +
                 's until timeout')+'),\t',
        print title_txt+'total: '+str(round(100*trel, 1))+'% cycle[' + \
            str(step)+']: '+str(round(100*crel, 1))+'%'
        if timeout[0]:
            t = time.time()
            cycletime = ((2**step+1)**2-tsp[0]) * tdiff / float(tsp[0])
            print '\nTimeout reached!\nDo you want to stop, complete this' + \
                ' cycle (estm. '+str(round(cycletime, 1))+'s) or continue ' + \
                'by specifing an additional timeout (estm. total=' + \
                str(round(total-tdiff, 1))+'s)'
            rp1 = True
            while rp1:
                expr = raw_input('stop, complete cycle, new timeout ' +
                                 '[a,b,c]: ')
                if expr in ['', 'a', 'A']:
                    print 'stopping...'
                    abort[0] = True
                    rp1 = False
                elif expr in ['b', 'B']:
                    print 'completing cycle...'
                    timeout[0] = False
                    add2timeout[0] = round(cycletime*2)
                    stopafter[0] = True
                    rp1 = False
                elif expr in ['c', 'C']:
                    rp2 = 0
                    while rp2 < 3:
                        expr = raw_input('additional time for computing: ')
                        try:
                            add2timeout[0] += float(expr)
                            timeout[0] = False
                            break
                        except ValueError as p:
                            rp2 += 1
                            print p, 'retry (aborting after', 3-rp2, \
                                'additional tries)'
                    if rp2 >= 3:
                        abort[0] = True
                    rp1 = False
                else:
                    print 'Unknown expression: "'+expr+'"'
            userwait_time[0] += time.time() - t

    def initializition():
        if trace:
            cam.start_tracing()
            cam.trigger()
        return [0, cam.grab(), 0]

    def do(arg1):
        move = move_methode(arg1[1])
        xmot.move(move[0]*cam.sensor_pixel_height)
        zmot.move(move[1]*cam.sensor_pixel_width)  # .wait()
        cam.trigger()
        arg1[0] += 1
        arg1[1] = cam.grab()
        arg1[2] = arg1[2]+1 if (move**2).sum() < (tolerance/10.)**2 else 0
        return arg1

    def dest_reached(arg1):
        coord = cam2pixel()
        return ((coord - beamcenter)**2).sum() < tol2

    def com_at_frmcenter(arg1):
        img = arg1[1]
        com = center_of_mass(img-img.min())
        frm_center = np.array(img.shape) - 0.5
        return ((com-frm_center)**2).sum() < tol2

    def abort_test(arg1):
        no_move = arg1[2] >= 3
        r = cam2pixel()
        outbound = ((r < 0).any()) or ((r >= frm_shape).any())
        return no_move or outbound

    def test_sp(y, x):
        xmot._position = 0
        zmot._position = 0
        pixel2cam([y, x])
        _iter = [0]
        #res = converge([do], dest_reached, max_iter, abort_test,
        #               initializition, None, _iter)
        res = converge([do], com_at_frmcenter, max_iter, None,
                       initializition, None, _iter)
        count = int(_iter[0][0]*255./max_iter)
        if res == 'success':
            v1 = 'dr_s'
            v2 = 3
            c_s[0] += 1
        elif res == 'fail':
            v1 = 'dr_f'
            v2 = 2
            c_f[0] += 1
        else:
            v1 = 'dr_a'
            v2 = 1
            c_a[0] += 1
        col = lcolor(v1, count)
        img_dr[y, x] = col
        dot = np.array([[-1, 0, 0, 0, 1], [0, -1, 0, 1, 0]])
        if trace:
            from __new__test import line
            tr = cam.stop_tracing()
            tr = map(cam2pixel, tr)
            d = dot + tr[0].reshape(2,1)
            img_tr[d[0], d[1]] = col
            for i in range(len(tr)-1):
                line(img_tr, tr[i], tr[i+1], col)
        epy, epx = cam2pixel()
        if (epy > 0) and (epy < frm_shape[0]) and \
                (epx > 0) and (epx < frm_shape[1]):
            img_ep[epy, epx] += 1
            epm[epy, epx] = v2 if v2 > epm[epy, epx] else epm[epy, epx]
        else:
            outcount[0] += 1
        tsp[0] += 1

    # init dummys
    cam = DummyCamera()
    o_sh = cam._imgshape
    if isinstance(cam_data, str) and len(cam_data) > 0:
        from scipy.ndimage import imread
        img = imread(cam_data)
        data_source = cam_data
    elif isinstance(cam_data, dict):
        img = cam_data['img']
        data_source = cam_data['img_name']
    elif isinstance(cam_data, np.ndarray):
        img = cam_data
        data_source = 'given image'
    else:
        cam._imgshape = (6000, 8000)
        cam._sigma = 1000
        cam._rel_sec = 0.4
        cam._m_ax = [1, 0]
        cam.trigger()
        img = cam.grab()
        data_source = 'generated'
    cam.from_image(img)
    cam._imgshape = o_sh
    frm_shape = np.array(img.shape)
    cam._noise = imgnoise
    xmot = DummyMotor(cam, [1, 0])
    zmot = DummyMotor(cam, [0, 1])

    # init
    bc = center_of_mass(img-img.min())
    pixel2cam(bc)
    converge([do], lambda x: False, 10, abort_test, initializition)
    beamcenter = cam2pixel()

    tol2 = tolerance ** 2
    title_txt = 'Running test: '
    outcount = [0]
    c_s = [0]
    c_f = [0]
    c_a = [0]
    tsp = [0]
    hist = {}
    epm = np.zeros(frm_shape, np.uint8)
    spm = np.zeros(frm_shape, np.bool_)
    start = np.round(np.array(cam._imgshape)/2)
    length = np.array(frm_shape)-cam._imgshape
    spm[start[0]:start[0]+length[0]+1, start[1]:start[1]+length[1]+1] = True
    userwait_time = [0]
    add2timeout = [0]
    timeout = [False]
    stopafter = [False]
    abort = [False]

    # init result-images
    img_dr = np.require(np.tile(lcolor('bg'), frm_shape), np.uint32)
    img_dr[start[0]:start[0]+length[0]+1, start[1]:start[1]+length[1]+1] = \
        lcolor('u_sp')
    img_ep = np.zeros(frm_shape, np.uint32)
    img_tr = img_dr.copy()

    # testloop
    step = -1
    orel = -1
    old_time = time.time()
    trel = 0
    tstep = 1./((length+1).prod())
    while (spm.any()) and (not abort[0]) and (not stopafter[0]):
        step += 1
        y = 0.
        ystep, xstep = np.require(length, np.float64)/2**step
        crel = 0.
        cstep = 1/4 if step == 0 else 1/(3./4*(2**step)**2+2**step)

        while y <= length[0] and (not abort[0]):
            x = 0.

            while x <= length[1] and (not abort[0]):
                ry, rx = np.round([y, x])+start
                if spm[ry, rx]:
                    test_sp(ry, rx)
                    spm[ry, rx] = False
                    trel += tstep
                    crel += cstep
                x += xstep

                tdiff = time.time()-old_time-userwait_time[0]
                timeout[0] = (tdiff > max_time+add2timeout[0]) and \
                    (not stopafter[0])
                if timeout[0] and (not abort[0]):
                    print_progress()

            y += ystep
            if orel < round(1000*trel) and (not abort[0]):
                print_progress()
                orel = round(1000*trel)

        if not abort[0]:
            print_progress()

    print 'computed '+str(tsp[0])+' startpoints in ' + \
        str(round(time.time()-old_time-userwait_time[0], 1))+'s'

    # visualize data
    smax, fmax, amax = 0, 0, 0
    val = np.unique(epm)
    for i in val[val != 0]:
        m = epm == i
        max_ = img_ep[m].max() if img_ep[m].size > 0 else 0
        if i == 3:
            #print 'smax =', max_
            smax = max_
            c = 'ep_s'
        elif i == 2:
            #print 'fmax =', max_
            fmax = max_
            c = 'ep_f'
        elif i == 1:
            #print 'amax =', max_
            amax = max_
            c = 'ep_a'
        func = lambda x: lcolor(c, (x-1)*255./max_)
        cmap = np.array(map(func, range(max_+1)), np.uint32)
        img_ep[m] = cmap[img_ep[m]]

    img_t = img.dtype
    if img_t != np.uint32:
        img = np.require(img, np.uint32)
        if img_t == np.int8 or img_t == np.uint8:
            img *= 2**16 + 2**8 + 1
    bs = np.floor(beamcenter - tolerance)
    be = np.floor(beamcenter + tolerance + 1)
    sl1, sl2 = slice(bs[0], be[0]), slice(bs[1], be[1])
    mask = ((np.indices(np.ceil([tolerance*2+1, tolerance*2+1])) +
            (beamcenter % 1).reshape(2, 1, 1)-tolerance)**2).sum(0) < tol2
    img[sl1, sl2] = np.where(mask, lcolor('bc_inraw'), img[sl1, sl2])
    col_usp = lcolor('u_sp')
    col_bc = lcolor('bc_indata')
    m1 = np.logical_and(mask, img_dr[sl1, sl2] == col_usp)
    img_dr[sl1, sl2] = np.where(m1, col_bc, img_dr[sl1, sl2])
    tmp = np.require(np.tile(lcolor('bg'), frm_shape), np.uint32)
    tmp[start[0]:start[0]+length[0]+1, start[1]:start[1]+length[1]+1] = col_usp
    img_ep = np.where(epm == 0, tmp, img_ep)
    m2 = np.logical_and(mask, img_ep[sl1, sl2] == col_usp)
    img_ep[sl1, sl2] = np.where(m2, col_bc, img_ep[sl1, sl2])
    m3 = np.logical_and(mask, img_tr[sl1, sl2] == col_usp)
    img_tr[sl1, sl2] = np.where(m3, col_bc, img_tr[sl1, sl2])

    # result
    data = {'img': img, 'img_dr': img_dr, 'img_ep': img_ep,
            'ds': data_source, 'view': cam._imgshape, 'noise': imgnoise,
            'tol': tolerance, 'mi': max_iter, 'bc': beamcenter, 'cbc': bc,
            'oc': outcount[0], 'c_s': c_s[0], 'c_f': c_f[0], 'c_a': c_a[0],
            'tsp': tsp[0], 'ep_fmax': fmax, 'ep_smax': smax, 'ep_amax': amax}
    if trace:
        data['img_tr'] = img_tr

    return data


def converge_test_save_result(testdata, filename='converge_test.npz'):
    """"""

    td = testdata
    td['img'] = np.require(td['img'], np.dtype('<u4'))
    td['img_dr'] = np.require(td['img_dr'], np.dtype('<u4'))
    td['img_ep'] = np.require(td['img_ep'], np.dtype('<u4'))
    try:
        td['img_tr'] = np.require(td['img_tr'], np.dtype('<u4'))
    except KeyError:
        pass
    np.savez_compressed(filename, **td)


def converge_test_load_result(filename='converge_test.npz'):
    """"""

    tmp = np.load(filename)
    data = {}
    for x in tmp:
        data[x] = tmp[x]
    data['img'] = data['img'].view(np.dtype('<u4'))
    data['img_dr'] = data['img_dr'].view(np.dtype('<u4'))
    data['img_ep'] = data['img_ep'].view(np.dtype('<u4'))
    try:
        data['img_tr'] = data['img_tr'].view(np.dtype('<u4')) 
    except KeyError:
        pass

    return data


def converge_test_view_result(testdata):
    """Visualizes the testdata."""

    def color_expl_txt(prefix):
        e1, e2 = [td['ep_smax'], td['ep_amax']] if prefix == 'ep_' else \
            [td['mi'], td['mi']-1]
        return 'colors:\n  backgroundcolor: '+lcol_txt('bg') + \
            '\n  pixels not used as startingpoint: '+lcol_txt('u_sp') + \
            '\n  beamcenter-mark: '+lcol_txt('bc_indata') + \
            '\n  success: '+lcol_txt(prefix+'s', 1, e1) + \
            '\n  fail: '+lcol_txt(prefix+'f') + \
            '\n  abort: '+lcol_txt(prefix+'a', 1, e2)

    td = testdata
    stats_txt = 'statistics:' + \
        '\n  total number of startingpoints: '+str(td['tsp']) + \
        '\n  destination reached: '+str(td['c_s'])+' startpoints (' + \
        str(round(100*td['c_s']/float(td['tsp']), 1))+'%)' + \
        '\n  failed (after '+str(td['mi'])+' iterations): '+str(td['c_f']) + \
        ' startpoints ('+str(round(100*td['c_f']/float(td['tsp']), 1))+'%)' + \
        '\n  aborted: '+str(td['c_a'])+' startpoints (' + \
        str(round(100*td['c_a']/float(td['tsp']), 1))+'%)'
    img = td['img']

    i1 = {'ndArr': img, 'imgInfo': 'Mainimage ('+str(td['ds'])+', ' +
          str(img.shape[1])+'x'+str(img.shape[0])+'pixels) that is used as ' +
          'a template.\nThe viewport of the cutout image is ' +
          str(td['view'][1])+'x'+str(td['view'][0])+'pixels.'+'\nnoise = ' +
          str(td['noise'])+'\ntolerance = '+str(td['tol'])+'pixels' +
          '\nmax_iterations = '+str(td['mi'])+'\ncenter of mass of the ' +
          'template is [x: '+str(td['cbc'][1])+', y: '+str(td['cbc'][0]) +
          '] and\ncenter of mass + converge (10iterations) is [x: ' +
          str(td['bc'][1])+', Y: '+str(td['bc'][0])+'] (marked ' +
          colors['bc_inraw']['txt']+')\n\n'+stats_txt}
    i2 = {'ndArr': td['img_dr'], 'imgInfo': 'Image indicating for each pixel' +
          ' as a startingpoint if the beamcenter could be reached (within ' +
          'tolerance) after '+str(td['mi'])+' iterations.\n\n' +
          color_expl_txt('dr_')+'\n\n'+stats_txt}
    i3 = {'ndArr': td['img_ep'], 'imgInfo': 'Image indicating the endpoints ' +
          'after moving towards the center of mass.\n\n' +
          color_expl_txt('ep_')+'\n\n'+stats_txt+'\n'+str(td['oc']) +
          ' startpoints left this area'}
    tabs = [{'tabTitle': 'cam_rawdata', 'images': i1},
            {'tabTitle': 'destination reached', 'images': i2},
            {'tabTitle': 'endpoints', 'images': i3}]

    try:
        i4 = {'ndArr': td['img_tr'], 'imgInfo': 'Image with traces for each'+
              'startpoint.\n\n'+color_expl_txt('dr_')+'\n\n'+stats_txt}
        tabs.append({'tabTitle': 'trace', 'images': i4})
    except KeyError:
        pass

    i4 = {'ndArr': img, 'imgTitle': 'raw_data'}
    i5 = {'ndArr': td['img_dr'], 'imgTitle': 'destination reached'}
    i6 = {'ndArr': td['img_ep'], 'imgTitle': 'endpoints'}
    tabs.extend([{'tabTitle': 'raw + dest_reach compare', 'images': [i4, i5]},
                 {'tabTitle': 'raw + endpoints compare', 'images': [i4, i6]}])

    fact = int(max(img.shape)/500)
    if fact > 1:
        i7 = {'ndArr': img[0:-1:fact, 0:-1:fact],
              'imgTitle': 'raw_data (resampled)',
              'imgInfo': i1['imgInfo']}

        sh = int(img.shape[0]/fact), int(img.shape[1]/fact)
        dtyp = np.dtype([('k1', 'u1'), ('k2', 'u1'),
                         ('k3', 'u1'), ('k4', 'u1')])

        dr_s = td['img_dr'][:fact*sh[0], :fact*sh[1]].copy().\
            reshape(sh[0], fact, sh[1], fact).swapaxes(1, 2)
        dr_s_bgra = dr_s.view(dtyp)
        dr_d = np.zeros(sh, np.uint32)
        dr_d_bgra = dr_d.view(dtyp)
        for y in range(sh[0]):
            for x in range(sh[1]):
                pix = dr_s[y, x]
                mask = np.logical_and(pix != lcolor('bg'),
                                      pix != lcolor('u_sp'))
                mask = np.logical_and(mask, pix != lcolor('bc_indata'))
                for k in ['k1', 'k2', 'k3', 'k4']:
                    px = dr_s_bgra[k][y, x]
                    pxm = px[mask]
                    dr_d_bgra[k][y, x] = px.mean() if pxm.size == 0 \
                        else pxm.mean()
        i8 = {'ndArr': dr_d, 'imgTitle': 'dest_reached',
              'imgInfo': i2['imgInfo']}

        ep_s = td['img_ep'][:fact*sh[0], :fact*sh[1]].copy().\
            reshape(sh[0], fact, sh[1], fact).swapaxes(1, 2)
        ep_s_bgra = ep_s.view(dtyp)
        ep_d = np.zeros(sh, np.uint32)
        ep_d_bgra = ep_d.view(dtyp)
        for y in range(sh[0]):
            for x in range(sh[1]):
                pix = ep_s[y, x]
                mask = np.logical_and(pix != lcolor('bg'),
                                      pix != lcolor('u_sp'))
                mask = np.logical_and(mask, pix != lcolor('bc_indata'))
                for k in ['k1', 'k2', 'k3', 'k4']:
                    px = ep_s_bgra[k][y, x]
                    pxm = px[mask]
                    ep_d_bgra[k][y, x] = px.mean() if pxm.size == 0 \
                        else pxm.mean()
        i9 = {'ndArr': ep_d, 'imgTitle': 'endpoints', 'imgInfo': i3['imgInfo']}
        tabs.append({'tabTitle': 'overview', 'images': [i7, i8, i9]})

    w = {'windowTitle': 'Converge-Test', 'tabs': tabs}
    return ImVis(w)


















