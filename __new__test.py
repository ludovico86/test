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
    
    diff = stop[0]-start[0], stop[1]-start[1]
    steps = round(max(abs(diff[0]),abs(diff[1])))
    if steps == 0:
        return
    inx = diff[0]/steps
    iny = diff[1]/steps
    if linestyle == None or len(linestyle) == 0:
        l = range(int(steps)+1)
    else:
        from math import sqrt
        
        sl = sqrt(inx**2 + iny**2)
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
    for b in pix:
        exec('ndArr[b[0], b[1]]' + mode + str(color))

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
    
    block_mean[xf, yf] = nda[xf, ..., yf, ...].reshape(ns.prod(), bs**2).mean(axis=1)
    block_std[xf, yf] = nda[xf, ..., yf, ...].reshape(ns.prod(), bs**2).std(axis=1)
    
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











