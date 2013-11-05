"""
Beam_Centering #TODO complete documentation
"""

import concert.asynchronous as async
async.DISABLE = True

import numpy as np
from concert.quantities import q
from concert.base import Parameter
from concert.devices.cameras.base import Camera
from concert.devices.motors.base import Motor
from imwin import ImVis


class UnableToCenter(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def disk_2d(radius):
    """
    disk_2d creates a bool-type 2d-ndArray with round(radius*2+1)^2-Elements
    and True-Values for length PixelPos->Center <= Radius.
    Used as Structurelement for morpholocal-operations.
    """
    
    r = round(radius*2+1)
    a = np.zeros((r,r), dtype=np.bool_)
    r2 = (radius**2)
    cent = (a.shape[0]-1)/2
    for x in range(a.shape[0]):
        x2 = (cent-x)**2
        for y in range(a.shape[1]):
            if (x2 + (cent-y)**2) <= r2:
                a[x,y] = True
    return a


def object_count(frame, mf_size=3, thres=0.5, cl_size=2.6, op_size=5.2,
        lbl_struct=np.ones((3,3)), mfiltered=None, binarized=None, closed=None,
        opened=None, labeled=None):
    """Counts visible Objects after processing the frame.
    
    Processsteps:
        Median-Filter
        Binarization
        Binary-Closing
        Binary-Opening
        Labeling
        
    Optional Parameter:
    mf_size         Size of the square used to median_filter the frame.
                    (default: 3)
    thres           Threshold for binarization.
                    (default: 0.5; meaning: 0.5*(max-min)+min))
    cl_size         Radius (float possible) of the disk used for binary_closing.
                    (default: 2.6)
    op_size         Radius (float possible) of the disk used for binary_opening.
                    (default: 5.2)
    lbl_struct      Symmetric structure indicating neighbor-pixels.
                    (default: np.ones((3,3)))
    mfiltered       ndarray of same size as frame that will hold the median-
                    filtered frame afterwards.
    binarized       ndarray of same size as frame that will hold the binarized
                    frame afterwards.
    closed          ndarray of same size as frame that will hold the frame
                    after morpholocal closing.
    opened          ndarray of same size as frame that will hold the frame
                    after morpholocal opening.
    labeled         ndarray of same size as frame that will hold the labeled
                    frame afterwards.
    """
    
    from scipy.ndimage.morphology import binary_opening, binary_closing
    from scipy.ndimage.measurements import label
    from scipy.ndimage.filters import median_filter
    from math import ceil
    
    mframe = median_filter(frame, mf_size)
    mi, ma = mframe.min(), mframe.max()
    binary = (mframe > (thres*(ma-mi)+mi))
    c = ceil(cl_size)
    c1, c2 = [c+x for x in frame.shape]
    blobs1 = np.zeros([x+int(2*ceil(cl_size)) for x in frame.shape])
    #Working with included border to minimaze effects of the edge during
    #morpholocal transformation.
    blobs1[c:c1, c:c2] = binary
    blobs1 = binary_closing(blobs1, disk_2d(cl_size))
    blobs2 = binary_opening(blobs1, disk_2d(op_size))
    blobs1 = blobs1[c:c1,c:c2]
    blobs2 = blobs2[c:c1,c:c2]
    lbl, num_l = label(blobs2, lbl_struct)
        
    if isinstance(mfiltered, np.ndarray):
        mfiltered.flat[...] = mframe.flat[...]
    if isinstance(binarized, np.ndarray):
        binarized.flat[...] = binary.flat[...]
    if isinstance(closed, np.ndarray):
        closed.flat[...] = blobs1.flat[...]
    if isinstance(opened, np.ndarray):
        opened.flat[...] = blobs2.flat[...]
    if isinstance(labeled, np.ndarray):
        labeled.flat[...] = lbl.flat[...]
        
    return num_l


def beam_centering(cam, xmotor, zmotor, tolerance=5.0, max_iterations=10,
        move_axes=None, silent=False, debug=None):
    """beam_centering centers the beam visible through "cam" (cameradevice)
    using xmotor and zmotor to move the camera.
    Returns true on success and false on fail.
    
    tolerance:
        beam_centering tries to reduces the distance from beamcenter to 
        screencenter lower then tolerance (in pixels).
        Caution: tolerance is also used to determine if the beam has moved
        after giving the motors a move instruction during axis-finding. A
        tolerance too big, might prevent the function from identifying
        misaligned axes.
    
    max_iterations:
        Determines the maximum number of moves used to try to center the
        beam (not counting tries used to determine the moving-axis of the
        motors (6 at max per try)).
    
    move_axes:
        List of motor-movement-axes. default [xmotor-axis, zmotor-axis]
        xmotor-axis = [x-comp, y-comp]
        => letting the xmotor move by 1pixel-width lets the beam
           actually move "x-comp"-pixels in x-direction, and "y-comp"-pixels
           in y-direction
        beam_centering calculates the invers-matrix of these axes to
        transform pixel-information to motor-instructions.
                     
    silent:
        Deactivates generall print-outs and Exceptions in case of fail.
                     
    debug:
        Optional list-variabel that afterward holds information on all
        steps in a list. Each entry is dict-variabel with following keys:
        frame                   = data from cam (ndarray)
        mask                    = binary-mask indicatingdetection of beam
                                  (ndarray)
        beam_visible            = True if beam detected on screen
        beam_at_border          = True if beam touches the border of screen
        beam_coordinate         = center of mass coordinates of the beam on
                                  screen ndarray([x-comp., y-comp.])
        deviation_from_center   = deviation of the beam from center of screen
                                  {'x':x-div,'y':y-div,'dist':distance in pixels}
        motor_coordinate        = motorposition at that step ndarray([xmot,zmot])
        move_axes               = vectors of direction of each motor relativ to
                                  the screen
                                  ndarray([[xmot.x,xmot.y],[zmot.x,zmot.y]])
        step_info               = information on what is done in this step
        result                  = only on last list-entry!
                                  the result of the function as a string
                                  (success or failure, times of iteration)
    """
    
    def to_db(db, key, val, set_method='='):
        if debug != None:
            exec 'db[-1][key]'+set_method+'val' in locals()
            #pass
    
    def next_step(step_info='', iteration=False, find_axes_iteration=False):
        """Procedure to ensure proper debug-information handling."""
        gv['step'] += 1
        if iteration:
            gv['iter'] += 1
        if find_axes_iteration:
            gv['fa_iter'] += 1
        db.append({'beam_visible':None,
                   'beam_at_border':None,
                   'beam_coordinate':None,
                   'deviation_from_border':None,
                   'deviation_from_center':None,
                   'motor_coordinate':None})
        if debug != None:
            db[-1]['frame'] = 0
            db[-1]['mask'] = 0
            db[-1]['move_axes'] = ''
            db[-1]['step_info'] = step_info
    
    def get_data():
        """Get all information from the cam for the current step."""
        # Grab a frame
        cam.start_recording()
        cam.trigger()
        frm = cam.grab()
        gv['fshape'] = frm.shape
        to_db(db, 'frame', frm)
        #if gv['step'] == 0:
        #    db[0]['win'] = ImVis(np.require(frm/256, np.uint8))
        #else:
        #    db[-1]['win'] = db[0]['win'].add_image(np.require(frm/256, np.uint8))
        
        # Detect beam on screen
        mfrm = np.zeros(frm.shape)
        lbl = np.zeros(frm.shape)
        obj_c = object_count(frm, mfiltered=mfrm, labeled=lbl)
        bv = obj_c == 1
        gv['bv'] = bv
        db[-1]['beam_visible'] = bv
        to_db(db, 'mask', lbl != 0)
        
        # Detect if beam touches the border
        bab = any([any(lbl[:,0]), any(lbl[:,-1]), any(lbl[0,:]), any(lbl[-1,:])])
        db[-1]['beam_at_border'] = bab
        
        # Calculate center of mass of the beam
        from scipy.ndimage.filters import gaussian_filter
        filt = gaussian_filter(lbl, 20)
        total = filt.sum(dtype='f')
        bc = np.array([-1,-1] if total == 0 else \
                [np.arange(filt.shape[0]).dot(filt.sum(1)/total),\
                 np.arange(filt.shape[1]).dot(filt.sum(0)/total)])
        db[-1]['beam_coordinate'] = bc
        
        #Calculate deviation from border
        db[-1]['deviation_from_border'] = np.array([bc, \
                np.array(frm.shape)-bc]).min()
        
        # Calculate deviation from center
        diff = (np.array(frm.shape)-1)/2. - bc
        gv['diff'] = diff
        dist = np.sqrt((diff**2).sum())
        gv['dist'] = dist
        to_db(db, 'deviation_from_center', {'x':diff[0],'y':diff[1],'dist':dist})
    
    def beam_at_center():
        """Checks if distance to center is smaller then tolerance"""
        return (gv['dist'] <= tolerance) and db[-1]['beam_visible']
    
    def set_axes(ax):
        """Calculates the inverse of ax to be able to calculate
        motor-move-instructions from image-coordinates"""
        gv['move_axes'] = np.array(ax)
        try:
            gv['inv_axes'] = np.linalg.inv(ax)
        except np.linalg.LinAlgError:
            to_db(db, 'step_info', '\nnumpy.linalg.LinAlgError: singular matrix'+\
                  '\naxes identical or one equal to zeros', '+=')
    
    def along_axes():
        """Calculates move-instructions from diff (image-coordinate-system)
        using the motor-coordinate-system"""
        return gv['inv_axes'].T.dot(gv['diff'])
    
    def move_motors(xpos=None, zpos=None, xmove=None, zmove=None,
                    *args, **kwargs):
        """Moves the motors.
        First the motors are moved to position, afterwards they are moved
        relative. If neither position nor move is set, the procedure tries
        to center the beam using information from gv['diff'] and
        gv['inv_axes']."""
        if np.equal([xpos, zpos, xmove, zmove], None).all():
            it = kwargs.pop('iteration', True)
            st = kwargs.pop('step_info', 'iteration: '+str(gv['iter']+1))
            next_step(st, it, *args, **kwargs)
            x, y = along_axes()
            xmotor.move(x * mfact[0])
            zmotor.move(y * mfact[1]).wait()
            to_db(db, 'move_axes', gv['move_axes'])
        else:
            next_step(*args, **kwargs)
            if xpos != None:
                print 'set xpos to', xpos
                xmotor.position = xpos
            if zpos != None:
                print 'set zpos to', zpos
                zmotor.position = zpos
            if xmove != None:
                print 'xmove to', xmove
                xm = xmotor.move(xmove*mfact[0])
                if zmove == None:
                    xm.wait()
            if zmove != None:
                print 'zmove to', zmove
                zmotor.move(zmove*mfact[1]).wait()
        
        db[-1]['motor_coordinate'] = np.array([xmotor.position,zmotor.position])
        get_data()
        if gv['step']==100: exit() #TODO remove after debug
    
    def step_back(step, step_info, to_best_previous=False):
        """Returns to a previous position (after moving the beam out of 
        the field of vision or fail to find axes)"""
        if to_best_previous:
            #TODO
            pass
        else:
            print 'step_back('+str(step)+('-1' if step<0 else '')+')',
            m = db[step if step > 0 else step-1]['motor_coordinate']
            print 'coord: now=', db[-1]['motor_coordinate'], 'back to', m
            move_motors(m[0], m[1], step_info=step_info)
            
            #xmotor.position = m[0]
            #zmotor.position = m[1]
            #next_step('undone last move',iteration)
            #db[-1]['motor_coordinate'] = m
            to_db(db, 'move_axes', 'axes seem to be wrong')
            #get_data()
    
    def find_axes():
        """After trying to center the beam using the assumed
        coordinate-system failed, try to determine the coordinate-system"""
        
        def find_this_ax(n_ax):
            """Try to move motor[n_ax] to determine its axis of movement"""
            step = [min(gv['fshape'])/10.] *2
            old_iterstep = gv['step']
            old_bc = db[-1]['beam_coordinate']
            old_mc = db[-1]['motor_coordinate']
            diff = [np.zeros(2),np.zeros(2)]
            sstep = [0,0]
            
            laststep = ''
            count = 6
            while count > 0:
                count -= 1
                txt = fa_txt+'\ntry moving '+n[n_ax]+' by '+\
                      str(step[0]*mfact[n_ax])
                k = {('xmove' if n_ax==0 else 'zmove'):step[1]}
                move_motors(step_info=txt, find_axes_iteration=fa_upcount[0], 
                            **k)      
                #next_step(, False, fa_upcount[0])
                fa_upcount[0] = False
                #m[n_ax].move(step[1]*mfact[n_ax]).wait()
                #db[-1]['motor_coordinate'] = np.array([m[0].position,\
                #                                       m[1].position])
                #bv = get_data()
                bab = db[-1]['beam_at_border']
                bc = db[-1]['beam_coordinate']
                
                if not gv['bv']:
                    to_db(db, 'step_info', '\nfailed: beam not visible', '+=')
                    if laststep == 'inv':
                        step = [-step[0]/10, -1.1*step[0]]
                        print fa_txt+' '+str(step)+' (inverse)'
                        laststep = 'down'
                    else:
                        step = [-step[0], -2*step[0]]
                        print fa_txt+' '+str(step)
                        laststep = 'inv'
                    continue
                
                d = bc - old_bc
                if (d**2).sum() <= tolerance**2:
                    to_db(db, 'step_info','\nfailed: no movement detected','+=')
                    step = [10*step[0], 9*step[0]]
                    laststep = 'up'
                    continue
                
                if np.sqrt((d**2).sum()) > np.sqrt((diff[1]**2).sum()):
                    diff[1] = d
                    sstep[1] = step[0]
                if not bab:
                    diff[0] = d
                    sstep[0] = step[0]
                #elif bab and count > 0:
                #    to_db(db, 'step_info', 'try to not hit the border')
                #    step = [-step[0], -2*step[0]] if laststep == '' else \
                #           [-step[]]
                #           #TODO implement intelligent movement
                #    continue
                
                if any(diff[0] != 0):
                    ax[n_ax] = diff[0]/sstep[0]
                else:
                    ax[n_ax] = diff[1]/sstep[1]
                to_db(db, 'step_info', '\naxis of '+n[n_ax]+\
                      ': (x='+str(ax[n_ax][0])+\
                      ', y='+str(ax[n_ax][1])+')', '+=')
                return True
            
            step_back(old_iterstep, 'move back after failing to center')
            return False
        
        fa_txt = 'find_axes('+str(gv['fa_iter'])+'):'
        fa_upcount = [True]
        ax = [0,0]
        m = [xmotor, zmotor]
        n = ['xmotor', 'zmotor']
        
        x, y = (gv['fa_iter'] % 2), ((gv['fa_iter']+1) % 2)
        if not find_this_ax(x):
            x, y = y, x
            if not find_this_ax(x):
                return False
        if not find_this_ax(y):
            return False
        
        set_axes(ax)
        move_motors()
        return True
    
    def on_fail(txt):
        """Reaction on failure"""
        to_db(db, 'result', 'Failed: '+txt)
        if silent:
            return False
        else:
            raise UnableToCenter(txt)
    
    mfact = [cam.sensor_pixel_width, cam.sensor_pixel_height]
    gv = {'step':-1,'iter':-1,'fa_iter':0, 'move_axes':[], 'inv_axes':[],
          'fshape':[], 'diff':[], 'dist':0, 'bv':False} 
    db = [] if debug == None else debug
    
    next_step('initial state', True)
    db[-1]['motor_coordinate'] = np.array([xmotor.position, zmotor.position])
    set_axes([[1.,0.],[0.,1.]] if move_axes == None else move_axes)
    db[-1]['move_axes'] = gv['move_axes']
    
    get_data()
    while gv['iter'] < max_iterations and not beam_at_center():
        print "gv['step'] =", gv['step']
        if not gv['bv']:
            print "\nUnable to detected the beam on screen. gv['step'] =", \
                    gv['step'], '\n'
            if gv['iter'] == 0:
                return on_fail('Unable to detected the beam on screen.')
            step_back(-1, 'undone last move')
            print "\nfind_axes after beam lost\n"
            if not find_axes():
                return on_fail('Unable to determine axes of the motors.')
        elif gv['iter'] < 2:
            move_motors()
        else:
            print "\nfind_axes\n"
            if not find_axes():
                return on_fail('Unable to determine axes of the motors.')
    
    if gv['iter'] >= max_iterations:
        return on_fail('Unable to center beam after '+str(gv['iter'])+\
                       ' iterations.')
    else:
        to_db(db, 'result', 'Centered after '+str(gv['iter'])+'iterations.')
        if not silent:
            print '\nbeam_centering:\nCentered after ' + str(gv['iter']) + \
                'iterations.'
        return True







def gauss_spot(shape, sigma, coord=[0,0], rel_sec=1., m_axis=[1,0]):
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
        return ((x**2)/float(sigma**2)).reshape(shape[0],1) + \
                (y**2)/float(sigma**2)
    
    def ellipse2():
        vecs = np.indices(shape)
        vecs.shape = (2,1,shape[0]*shape[1])
        vecs[0,0,...] -= coord[0]
        vecs[1,0,...] -= coord[1]
        vecs = vecs.repeat(2, 1)
        sh = sigma.shape
        b = [sigma.flat[x] for x in range(sigma.size)]
        b = b * (shape[0]*shape[1])
        s = np.array(b)
        s.shape = (shape[0]*shape[1],sh[1],sh[0])
        s = s.swapaxes(0,2)
        sv = s * vecs
        vsv = vecs.swapaxes(0,1) * sv
        return np.sum(np.sum(vsv, 0), 0).reshape(shape[0], shape[1])
    
    def ellipse3():
        vecs = np.append(np.indices(shape), np.ones(shape[0]*shape[1]))
        vecs.shape = (3,1,shape[0]*shape[1])
        vecs = vecs.repeat(3, 1)
        sh = sigma.shape
        b = [sigma.flat[x] for x in range(sigma.size)]
        b = b * (shape[0]*shape[1])
        s = np.array(b)
        s.shape = (shape[0]*shape[1],sh[1],sh[0])
        s = s.swapaxes(0,2)
        sv = s * vecs
        vsv = vecs.swapaxes(0,1) * sv
        return np.sum(np.sum(vsv, 0), 0).reshape(shape[0], shape[1])
        
    from numbers import Number
    if isinstance(sigma, Number):
        if (rel_sec==1):
            return np.exp(-0.5*circle())
        ax = np.array(m_axis, np.float64)
        ax /= np.sqrt((ax**2).sum())
        v = np.mat([ax,[-ax[1],ax[0]]])
        sigma *= sigma
        d = np.eye(2) * np.array([sigma, sigma*rel_sec])
        sigma = np.mat(v.T * d * v)
    else:
        sigma = np.mat(sigma)
    
    if sigma.shape == (2,2):
        d = np.linalg.det(sigma)
        if d <= 0:
            print 'gauss_spot: sigma defines not an ellipse'
        sigma = np.linalg.inv(sigma)
        return np.exp(-0.5*ellipse2())
    
    if sigma.shape == (3,3):
        d = np.linalg.det(sigma)
        if d <= 0:
            print 'gauss_spot: sigma defines not an ellipse'
        sigma = np.linalg.inv(sigma)
        return np.exp(-0.5*ellipse3())
    
    raise ValueError('sigma can only be a scalar, 2x2 or 3x3-array-like')


def noise(ndArr, add_noise=(0,0), mult_noise=(1,1), snp_noise=(0,0,0),
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


def test_beam_centering(testcase=0, tolerance=None, max_iterations=None,
        imgnoise=None):
    
    class DummyCamera(Camera):
        
        def __init__(self, pos=(0.,0.), imgsize=(640,480)):
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
            self._m_ax = [1,0]
            self._imgsize = imgsize
            self._noise = imgnoise
            
            self._coord = [pos[0] * q.meter, pos[1] * q.meter]
            self._coord_at_trigger = [pos[0] * q.meter, pos[1] * q.meter]
        
        def _record_real(self):
            pass
        
        def _stop_real(self):
            pass
        
        def _trigger_real(self):
            self._coord_at_trigger = self._coord
            
        def _grab_real(self):
            c = [(self._coord_at_trigger[0]/self.sensor_pixel_width).\
                to(q.dimensionless).magnitude, (self._coord_at_trigger[1]/self.
                sensor_pixel_height).to(q.dimensionless).magnitude]
            print 'DummyCam.grab(): coord', c
            time = self.exposure_time.to(q.s).magnitude
            img = gauss_spot(self._imgsize, self._sigma, c, self._rel_sec, 
                    self._m_ax)* time*1000*(2**16-1)
            if self._noise != None:
                noise(img, ins=True, **self._noise)
            img[img < 0] = 0
            img[img > 2**16-1] = 2**16-1
            return np.require(img, np.uint16)
       
    class DummyMotor(Motor):
        
        def __init__(self, cam, axis, calibration=None, limiter=None,
                position=0, hard_limits=None):
            super(DummyMotor, self).__init__(calibration, limiter)
            self._remoteCam = cam
            self._axis = axis
            self._position = position
            self._hard_limits = (-100, 100) if hard_limits==None else hard_limits
            
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
    
    def above_noise_threshold(ndArr, blocksize=5, percent2mean=0.1):
        bs = blocksize
        ns = np.array(ndArr.shape, np.uint)/int(bs)
        nda = ndArr[:ns[0]*bs,:ns[1]*bs].reshape(ns[0], bs , ns[1], bs)
        
        block_mean = np.empty(ns)
        block_std = np.empty(ns)
        ind = np.indices(ns)
        xf = ind[0].flat[...]
        yf = ind[1].flat[...]
        
        block_mean[xf, yf] = nda[xf, ..., yf, ...].\
                                reshape(ns.prod(), bs**2).mean(axis=1)
        block_std[xf, yf] = nda[xf, ..., yf, ...].\
                                reshape(ns.prod(), bs**2).std(axis=1)
        
        b2m = max(1, int(percent2mean*ns.prod()))
        mean = np.sort(block_mean, axis=None)[0:b2m].mean()
        std = block_std.mean()
        
        return mean+2*std
        #TODO test
        
    def filtered_mask(ndArr, thres=0.1):
        from scipy.ndimage.filters import gaussian_filter, median_filter
        
        frm = gaussian_filter(median_filter(ndArr, 3), 20)
        return frm > thres*frm.ptp()+frm.min()
    
    def _info(ndArr, bc, is_ax, try_ax):
        from __new__test import line
            
        line(ndArr, [ndArr.shape[0]/2,0], [ndArr.shape[0]/2,ndArr.shape[1]-1], \
             (127*256+127)*256+127, '=', [3,3])
        line(ndArr, [0,ndArr.shape[1]/2], [ndArr.shape[0]-1,ndArr.shape[1]/2], \
             (127*256+127)*256+127, '=', [3,3])
             
        if bc != None:
            bc = np.require(np.round([bc-[0,2],bc+[0,2],
                                      bc-[2,0],bc+[2,0]]), np.uint)
            line(ndArr, bc[0], bc[1], 255*2**16)
            line(ndArr, bc[2], bc[3], 255*2**16)
        
        col = [250*256*256, 250*256, (180*256+10)*256+120, (120*256+200)*256+10]
        ls = [None, None, [6,3], [6,3]]
        if try_ax != None and not isinstance(try_ax, str):
            is_ax.extend(try_ax)
        
        for x in range(len(is_ax)):
            stt = [ndArr.shape[0]/2,ndArr.shape[1]/2]
            stp = [stt[0]+100*is_ax[x][0], \
                   stt[1]+100*is_ax[x][1]]
            stp[0] = 0 if stp[0] < 0 else \
                    (ndArr.shape[0]-1 if stp[0] >= ndArr.shape[0] else stp[0])
            stp[1] = 0 if stp[1] < 0 else \
                    (ndArr.shape[1]-1 if stp[1] >= ndArr.shape[1] else stp[1])
            line(ndArr, stt, stp, col[x], '=', ls[x])
            
    def vis_debug(debug):
        
        def try_read(key, pre=True, as_string=True):
            try:
                if as_string:
                    k = (key + ': ' if pre else '') + str(step[key])
                else:
                    k = step[key]
            except KeyError:
                if as_string:
                    return ''
                else:
                    return None
            return k 
        
        from imwin import colorize
        tabs = []
        fi = True
        frame_size = (400,400)
        for step in debug:
            #print step
            if not isinstance(step['frame'], np.ndarray):
                print 'frame is not a np.ndarray (type='+str(type(step['frame']))+')'
                frm = np.zeros(frame_size)
            else:
                frame = step['frame']
                frame_size = frame.shape
                frm = colorize(frame, [255,255,255])
                frm_rgb = frm.view(np.dtype([('b','u1'),('g','u1'),('r','u1'),
                                             ('a','u1')]))
                frm_rgb['b'][filtered_mask(frame)] = 255
                #frm_rgb['r'][frame > above_noise_threshold(frame)] = 255
                #frm_rgb['g'][step['mask']] = 255
                frm_rgb['g'][step['mask']] = frm_rgb['g'][step['mask']]/2 + 128
            ma = try_read('move_axes', as_string=False)
            bc = try_read('beam_coordinate', as_string=False)    
            _info(frm, bc, [xmot._axis,zmot._axis], ma)
            k = {'ndArr':frm}
            k['imgInfo'] = try_read('beam_visible')+\
                           '\n'+try_read('beam_at_border')+\
                           '\n'+try_read('beam_coordinate')+\
                           '\n'+try_read('deviation_from_center')+\
                           '\n'+try_read('motor_coordinate')+\
                           '\nmove_axes:\n'+try_read('move_axes', False)+\
                           '\nmotor_axes(set by test):\n'+\
                           str(np.array([xmot._axis, zmot._axis]))+\
                           '\n'+try_read('step_info')
            tabs.append({'tabTitle':step['step_info'].split('\n')[0], \
                    'images':k})
            if fi:
                fi = False
                win_str = 'Beamcentering-Test: Testcase'+str(testcase)
                try:
                    win_str += '='+tc_str+'; '+debug[-1]['result']
                except KeyError:
                    pass
                win = {'windowTitle':win_str, 'tabs':tabs}
                debug[-1]['win'] = [ImVis(win)]
            else:
                debug[-1]['win'].append(debug[-1]['win'][0].add_image(tabs[-1]))
        
        k0 = {}
        k0.update(tabs[0]['images'])
        k0['imgTitle'] = 'initial state'
        k1 = {}
        k1.update(tabs[-1]['images'])
        k1['imgTitle'] = 'final state'
        tt = 'result'
        try:
            k1['imgInfo'] += '\n\nresult:\n'+debug[-1]['result']
        except KeyError:
            tt = 'Exception'
        debug[-1]['win'].append(debug[-1]['win'][0].add_image({'tabTitle':tt, \
                'images':[k0,k1]}))
    
    import random
    
    cam = DummyCamera()
    xmot = DummyMotor(cam, [1,0])
    zmot = DummyMotor(cam, [0,1])
    kwargs = {}
    if tolerance != None:
        kwargs['tolerance'] = tolerance
    if max_iterations != None:
        kwargs['max_iterations'] = max_iterations
    
    
    random.seed()
    if testcase < 0:
        pass
    if int(testcase) == 0:
        tc_str = 'Random beamsize(10...60sigma)/beamposition([0.1<=x<0.9]*framesize)'
        c =  [(0.1+0.8*random.random())*cam._imgsize[0]*cam.sensor_pixel_width,
              (0.1+0.8*random.random())*cam._imgsize[1]*cam.sensor_pixel_height]
        print 'test_beam_centering: start_coord', c
        cam._coord = c
        cam._sigma = 10 + random.random()*50
    if int(testcase) == 1:
        tc_str = 'Random beamsize(10...60sigma)/beamposition(on edge)'
        s = int(4*random.random())
        a, b, c = ((s//2)%2), ((s//2+1)%2), (s%2)
        r = random.random()
        c = [r*a+b*c, r*b+a*c]
        c[0], c[1] = c[0]*cam._imgsize[0]*cam.sensor_pixel_width, \
                     c[1]*cam._imgsize[1]*cam.sensor_pixel_height
        cam._coord = c
        cam._sigma = 10 + random.random()*50
    if testcase >= 2:
        tc_str = 'Random beamsize(10...60sigma)/beamposition/elliptic'
        c = [random.random()*cam._imgsize[0]*cam.sensor_pixel_width,
             random.random()*cam._imgsize[1]*cam.sensor_pixel_height]
        cam._coord = c
        cam._sigma = 10 + random.random()*50
        cam._rel_sec = 0.9*random.random()+0.1
        cam._m_ax = [random.random(), random.random()]
    if testcase >= 3:
        tc_str += '/axis-misalignment'
        v = 2*np.random.random(2)-1
        v /= np.sqrt((v**2).sum())
        print 'xmot._axis =', v
        xmot._axis = v
        v = np.array([2*random.random()-1,2*random.random()-1])
        v /= np.sqrt((v**2).sum())
        print 'zmot._axis =', v, '\n'
        zmot._axis = v
    if testcase >= 4:
        tc_str += '/motor_miscalibration'
        v = 10**(2*random.random()-1)
        print 'xmot._axis_length =', v
        xmot._axis *= v
        v = 10**(2*random.random()-1)
        print 'zmot._axis_length =', v
        zmot._axis *= v
    if testcase >= 5:
        cam._coord = [511*cam.sensor_pixel_width, 22*cam.sensor_pixel_height]
        cam._sigma = 60
        xmot._axis = [1.2243, -0.1268]
        zmot._axis = [-0.5980, -0.5824]
    
    if imgnoise == None:
        nl = testcase - int(testcase)
        cam._noise = {'add_noise':(-30000*nl,30000*nl),
                'mult_noise':(1-0.5*nl,1+0.5*nl),
                'snp_noise':(0.05*nl,0,2**16-1)}
    
    log = []
    try:
        d = beam_centering(cam, xmot, zmot, silent=False, debug=log, **kwargs)
        if d:
            print 'beam_centering: Successfully centered!'
        else:
            print 'beam_centering: UnableToCenter!'
    except:
        print 'exception in beam_centering'
        vis_debug(log)
        raise
    
    vis_debug(log)
    
    return log



