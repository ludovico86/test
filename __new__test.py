class BaseTest(object):
    pass

class Test(object):
    def __new__(cls, *args, **kwargs):
        if len(args) == 0 or args[0] == 1:
            print 'One instance of class Test created'
            return super(Test, cls).__new__(cls, *args, **kwargs)
        else:
            result = []
            c = args[0]
            args = args[1:]
            for x in range(c):
                result.append(super(Test, cls).__new__(cls, *args, **kwargs))
                result[-1].__init__(*args, **kwargs)
            print str(len(result))+' instances of class Test created'
            return result
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        print 'Instance '+str(self)+' initialized'
        print 'args: ', args
        print 'kwargs: ', kwargs
    def __del__(self):
        print str(self)+' deleted'



def myXPrint(ref, *args):
    for x in args:
        print str(x),
    print ' '


def object_tree(parentWindow):
    import weakref
    import imwin
    
    
    pw = parentWindow
    result = {'parentWindow':weakref.ref(pw,
            imwin._ext_Callback(myXPrint, 'parentWindow got killed.'))}
    for sync in pw._tablist.itervalues():
        obj = sync['ref']()
        result['Sync'+str(obj._id)] = weakref.ref(obj,
                imwin._ext_Callback(myXPrint, 'Sync', obj._id, ' got killed.'))
        for i in obj._syncList.itervalues():
            img = i['ref']()
            result['Sync'+str(obj._id)+'_Img'+str(img._id)] = weakref.ref(img,
                    imwin._ext_Callback(myXPrint, 'Img', img._id, ' got killed.'))
    return result


def instance_data_list(obj, objname=None):
    import weakref
    import imwin
    
    def cb(*args, **kwargs):
        args = list(args)
        args.append(' ('+str(count[0]-1)+' activ)')
        myXPrint(*args, **kwargs)
        count[0] -= 1
        if count[0] == 0:
            print objname+': all parameters deleted.'
    
    count = [0]
    if objname == None:
        objname = str(obj)
    result = {}
    for x,y in obj.__dict__.iteritems():
        try:
            result[x] = weakref.ref(y, imwin._ext_Callback(cb, 'obj='+objname+\
                    ': '+str(x)+' got deleted.'))
            count[0] += 1
        except:
            pass
            
    return result


def objTree_n_data(parentWindow):
    import weakref
    import imwin
    
    pw = parentWindow
    result = {}
    result['parentWindow'] = {'ref':weakref.ref(pw,
            imwin._ext_Callback(myXPrint, 'parentWindow got killed.'))}
    result['parentWindow']['inst_data'] = instance_data_list(pw, 'parentWindow')
    for sync in pw._tablist.itervalues():
        obj = sync['ref']()
        sync_name = 'Sync'+str(obj._id)
        result[sync_name] = {'ref':weakref.ref(obj,
                imwin._ext_Callback(myXPrint, sync_name+' got killed'))}
        result[sync_name]['inst_data'] = instance_data_list(obj, sync_name)
        for i in obj._syncList.itervalues():
            img = i['ref']()
            img_name = 'Img'+str(img._id)
            result[sync_name+'_'+img_name] = {'ref':weakref.ref(img,
                    imwin._ext_Callback(myXPrint, 'Img', img._id, ' got killed.'))}
            result[sync_name+'_'+img_name]['inst_data'] = instance_data_list(img,
                    sync_name+'_'+img_name)
    return result


def nicePrint_oTnd(data):
    print '\n\n'
    for n, d in data.iteritems():
        print n+':', 'ref:', d['ref']
        print n+':', 'inst_data:'
        
        for x, y in d['inst_data'].iteritems():
            print '   ', x, y
        
        print '\n'


from PyQt4 import QtCore

class _Communicate(QtCore.QObject):
    def __init__(self):
        super(_Communicate, self).__init__()
        self.none_arg = QtCore.pyqtSignal()
        self.one_arg = QtCore.pyqtSignal()
        self.multi_args = QtCore.pyqtSignal()
        

class Test2(object):
    
    def __init__(self, name='Instance of Test2'):
        super(Test2, self).__init__()
        self.name = name
        self._c = _Communicate()
        self._c.none_arg.connect(self._none_arg)
        self._c.one_arg.connect(self._one_arg)
        self._c.multi_args.connect(self._multi_args)
        
    def _none_arg(self):
        print 'I am', str(self)
        
    def _one_arg(self, n):
        print 'I am', 'what I am'*n
        
    def _multi_args(self, *args):
        for x in args:
            print str(x),
        print ' '


def line(ndArr, start, stop, color=1, mode='=', linestyle=None):
    import numpy as np
    
    diff = np.array([stop[0]-start[0], stop[1]-start[1]])
    steps = round(max(np.abs(diff)))
    if steps == 0:
        exec('ndArr[start[0], start[1]]'+mode+str(color))
        return
    inx, iny = diff/steps
    if linestyle == None or len(linestyle) == 0:
        l = range(int(steps)+1)
    else:
        sl = np.sqrt(inx**2 + iny**2)
        lss = np.array(linestyle).sum()
        linestyle = np.cumsum([0]+list(linestyle))
        l = [x for x in range(int(steps)+1) \
             if any([(((x*sl)%lss) >= linestyle[i-1] and \
                      ((x*sl)%lss) < linestyle[i]) \
                     for i in range(1,len(linestyle),2)])]
    
    p = np.array([[int(round(start[0]+inx*s)), int(round(start[1]+iny*s))] \
                      for s in l])
    pix = p[np.logical_and(np.logical_and(0 <= p[...,0], p[...,0] < ndArr.shape[0]),
                           np.logical_and(0 <= p[...,1], p[...,1] < ndArr.shape[1]))]
    exec('ndArr[pix[..., 0], pix[..., 1]]' + mode + str(color))

def line_test(linestyle=None):
    import numpy as np
    from imwin import ImVis
    
    a = np.zeros((401, 401), np.uint32)
    stt = [200, 200]
    stp = np.array([0, 0])
    go = np.array([[5, 0], [0, 5], [-5, 0], [0, -5]])
    for x in go:
        while ((stp+x) >= 0).all() and ((stp+x) < a.shape).all():
            stp += x
            line(a, stt, stp, 255*2**16, linestyle=linestyle)
    return ImVis(a)

def fixedlengthvec(vec, length):
    from math import sqrt
    
    l = 0
    if len(vec) == 2 and len(vec[0]) == 1:
        vec = [vec]
    for x in vec:
        b = sqrt(x[0]**2 + x[1]**2)
        if b > l:
            l = b
    result = []
    for x in vec:
        result.append([x[0]*length/l, x[1]*length/l])
    if len(result) == 1:
        return result[0]
    else:
        return result


def array_test():
    import numpy as np
    x = np.array([[[0], [1], [2]]])
    print x.shape
    print np.squeeze(x).shape
    y = x[..., np.newaxis]
    print y.shape


def modargs(arg0, arg1, defarg1=None, defarg2=None, debug=None, *args, **kwargs):
    if debug != None:
        debug['arg0'] = arg0
        debug['arg1'] = arg1
        debug['defarg1'] = defarg1
        debug['defarg2'] = defarg2
        for x in range(len(args)):
            debug['arg'+str(x+2)] = args[x]
        debug.update(kwargs)
    return debug != None


def inmodargs(*args, **kwargs):
    log = {}
    modargs(12, 23, *args, debug=log, **kwargs)
    if log.has_key('arg0'):
        print 'log erweitert'
    return log

def subextend():
    def sub(key, arg):
        gv[key] = arg
        
    gv = {}
    sub('key1', 'value1')
    sub('key2', 'value2')
    return gv


def above_noise_threshold(ndArr, blocksize=5, percent2mean=0.1):
    import numpy as np
    
    bs = blocksize
    ns = np.array(ndArr.shape, np.uint)/int(bs)
    nda = ndArr[:ns[0]*bs,:ns[1]*bs].reshape(ns[0], bs , ns[1], bs)
    
    block_mean = np.empty(ns)
    block_std = np.empty(ns)
    ind = np.indices(ns)
    xf = ind[0].flat[...]
    yf = ind[1].flat[...]
    
    block_mean[xf, yf] = nda[xf, ..., yf, ...].reshape(ns.prod(), bs**2).\
        mean(axis=1)
    block_std[xf, yf] = nda[xf, ..., yf, ...].reshape(ns.prod(), bs**2).\
        std(axis=1)
    
    b2m = max(1, int(percent2mean*ns.prod()))
    mean = np.sort(block_mean, axis=None)[0:b2m].mean()
    std = block_std.mean()
    
    return [mean+2*std, mean, std, block_mean, block_std]


def dist2border(frame_shape, coord):
    import numpy as np
    
    x = min(coord[0], frame_shape[0]-coord[0])
    y = min(coord[1], frame_shape[1]-coord[1])
    return (np.array([coord, frame_shape-coor]).min(0)**2).sum()
    return np.array([])


def pro_bc_test():
    import numpy as np
    from concert.quantities import q
    from processes import beam_centering
    from converge import DummyCamera, DummyMotor
    from scipy.ndimage import imread

    img = imread('stitched_image.jpg')
    xborder = [1.0*q.cm, 4.0*q.cm]
    zborder = [0.3*q.cm, 1.65*q.cm]
    cam = DummyCamera()
    o_sh = cam._imgshape
    sp = cam.sensor_pixel_height, cam.sensor_pixel_width
    r = {'img': img,
         'view': [(o_sh[x]*sp[x]).to(q.um.units).magnitude for x in [0,1]],
         'view_ip': o_sh}
    cam.from_image(img)
    coord = np.array([cam._imgshape[0]*sp[0], cam._imgshape[1]*sp[1]],
                     np.object) * [-0.15, 0.77]
    cam._imgshape = o_sh
    xmot = DummyMotor(cam, [0, 1])
    zmot = DummyMotor(cam, [1, 0])
    xmot.move(coord[1])
    zmot.move(coord[0])
    cam.start_tracing()
    cam.trigger()
    xb = [x.to(q.um.units).magnitude for x in xborder]
    zb = [z.to(q.um.units).magnitude for z in zborder]

    print 'start beam_centering'
    try:
        beam_centering(cam, xmot, zmot, sp, xborder, zborder, thres=30)
    except Exception as e:
        print 'Exception in beam_centering:', e
        r['exception'] = e

    tr = cam.stop_tracing()
    tr = [[c1.to(q.um.units).magnitude,
           c2.to(q.um.units).magnitude] for c1, c2 in tr]
    r['trace'] = tr
    r['border'] = [zb, xb]
    r['pixelsize'] = [x.to(q.um.units).magnitude for x in sp]
    print 'return from beam_centering'
    return r


def vis_bc_test(data):
    import numpy as np
    from concert.quantities import q
    from imwin import ImVis, colorize

    tr = np.array(data['trace'], np.float64)
    vw = np.array(data['view'], np.float64)
    bd = np.array(data['border'], np.float64)
    ps = data['pixelsize']
    img_0coord = np.array(data['img'].shape, np.float64)/2 * ps
    ny0 = -int(min(tr[:,0].min(), -img_0coord[0], bd[0, 0]))
    ny1 = int(max(tr[:,0].max(), img_0coord[0], bd[0, 1]))
    nx0 = -int(min(tr[:,1].min(), -img_0coord[1], bd[1, 0]))
    nx1 = int(max(tr[:,1].max(), img_0coord[1], bd[1, 1]))
    
    downscale = int(max(ny1+ny0, nx1+nx0, 1000)/1000)
    tr /= downscale
    vw /= downscale
    bd /= downscale
    img_0coord /= downscale
    ny0 = int(ny0/downscale)
    ny1 = int(ny1/downscale)
    nx0 = int(nx0/downscale)
    nx1 = int(nx1/downscale)

    mask1 = np.zeros(data['img'].shape, np.bool_)
    mask2 = mask1.copy()
    mask1[[x for x in np.arange(mask1.shape[0])*downscale/ps[0] \
           if x < mask1.shape[0]], :] = True
    s1 = mask1[:, 0].sum()
    mask2[:, [x for x in np.arange(mask2.shape[1])*downscale/ps[1] \
              if x < mask2.shape[1]]] = True
    s2 = mask2[0, :].sum()
    img = data['img'][np.logical_and(mask1, mask2)].reshape(s1, s2)
    
    #print 'ny: ['+str(ny0)+', '+str(ny1)+'], nx: ['+str(nx0)+', '+str(nx1)+']'
    overview = np.tile(np.array([(80*256+80)*256+80], np.uint32),
                       (ny1+ny0+41, nx1+nx0+41))
    ny0 += 20
    nx0 += 20
    n0 = np.array([ny0, nx0])
    sa_color = (50)*256+120
    overview[ny0+bd[0, 0]:ny0+bd[0, 1], nx0+bd[1, 0]:nx0+bd[1, 1]] = sa_color
    overview[ny0-img_0coord[0]:ny0-img_0coord[0]+img.shape[0],
             nx0-img_0coord[1]:nx0-img_0coord[1]+img.shape[1]] = \
        colorize(img, [255, 255, 255])
    bdf = np.require((bd + n0.reshape(2, 1)).flatten(), np.int)
    line(overview, bdf[[0, 2]], bdf[[1, 2]], sa_color)
    line(overview, bdf[[1, 2]], bdf[[1, 3]], sa_color)
    line(overview, bdf[[1, 3]], bdf[[0, 3]], sa_color)
    line(overview, bdf[[0, 3]], bdf[[0, 2]], sa_color)
    tr += n0
    print 'length of trace:', len(tr)
    vw_color = (255*256+200)*256
    vwp = np.array([[-vw[0]/2, -vw[1]/2], [vw[0]/2, -vw[1]/2],
                    [vw[0]/2, vw[1]/2], [-vw[0]/2, vw[1]/2]])
    for t in tr:
        line(overview, vwp[0]+t, vwp[1]+t, vw_color)
        line(overview, vwp[1]+t, vwp[2]+t, vw_color)
        line(overview, vwp[2]+t, vwp[3]+t, vw_color)
        line(overview, vwp[3]+t, vwp[0]+t, vw_color)
    for t in range(len(tr)-1):
        line(overview, tr[t], tr[t+1], 100*2**16)
        overview[round(tr[t][0]), round(tr[t][1])] = 255*2**16
        #print tr[t]
    overview[round(tr[-1][0]), round(tr[-1][1])] = 255*2**16

    k = {'ndArr': overview,
         'imgInfo': 'downscaled by '+str(downscale)+'x\nlength of trace: ' +
         str(len(tr))+'\ncam viewport: '+str(data['view'])+' in µm ' +
         str(data['view_ip'])+' in pixels'}
    try:
        k['imgInfo'] = '\n\nException: '+str(data['exception'])
    except KeyError:
        pass
    return ImVis({'windowTitle': 'beam_centering trace', 'tabs': k})


def test_decide_dir():
    import numpy as np

    def trydir(dr):
        print 'trydir('+str(dr)+')'
        print 'sp_pos =', sp_pos
        print 'n =', sp_pos + dir2rpos[dr]
        n = sp_pos + dir2rpos[dr]
        return 2 if not scanned[n[0], n[1]] else 0

    def decide_dir():
        """Picks a direction to go."""
        dr = np.array(map(trydir, range(4))) == 2
        print 'decide_dir: dr =', dr
        #print 'decide_dir: sp_shape =', sp_shape
        #print 'decide_dir: sp_pos =', sp_pos
        r = np.arange(4)[dr][0]
        print 'dir =', r
        w0 = sp_pos-[nz0, nx0]
        w1 = w0 + dir2rpos[r]
        print w0, w1
        w = (np.arctan2(w1[0], w1[1]) - np.arctan2(w0[0], w0[1])) % (2*np.pi)
        print 'w =', w
        search_rot[0] = int(w < np.pi)
        return r

    search_rot = [0]
    dir2rpos = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    scanned = np.zeros((11,21), np.bool_)
    nz0, nx0 = 5, 10
    sp_pos = np.array([3, 12])
    scanned[[3,4,4],[11,11,12]] = True
    print decide_dir()







