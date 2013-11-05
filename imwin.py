"""
ImageWindow displays an 2D-ndArray in grayscale using PyQt4.

e.g. a = np.random.random(640*480)*255
     a = a.reshape((640,480))
     a = np.require(a, np.uint8, 'C')
     win = ImageVisualizer(a, 'Randomvalues')
"""


import sys
import weakref
import numpy as np
from threading import Thread
from PyQt4 import QtGui, QtCore


class _ext_Callback():
    """Convenience-class for callbacks with additional arguments.
    
    Arguments safed during creation are appended to the list of arguments
    after the arguments the callback is called with.
    Keywordarguments from the call are updated with the saved ones, overriding
    the keywordarguments from the call with the same keyword.
    
    "credited to a Python news group posting by Timothy R. Evans"
    """
    
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs
            
    def __call__(self, *args, **kwargs):
        args = list(args)
        args.extend(self.args)
        kwargs.update(self.kwargs)
        return self.callback(*args, **kwargs)


class _Communicate(QtCore.QObject):
    def __init__(self):
        super(_Communicate, self).__init__()
        self.update_image = QtCore.pyqtSignal()
        self.deleteLater = QtCore.pyqtSignal()
    


class ImageVisualizer(object):
    """ImageVisualizer displays 2D-ndArrays using PyQt4.
    
    
    #TODO make it thread-safe
    (The application of all Qt-Widgets executes in a seperated thread.)
    
    example:
    
    a = np.array([[10,20,30],[40,50,60],[70,80,90]])
    
    """#TODO
    
    swapaxis = True
    
    class _Communicate(QtCore.QObject):
        update_image = QtCore.pyqtSignal()
        deleteLater = QtCore.pyqtSignal(list)
    
    def __new__(cls, *args, **kwargs):
        """Factory-function for creating one or multiple instances to display
        ndArrays"""
        
        def newInstance(cargs, ckwargs, iargs, ikwargs):
            temp = super(ImageVisualizer, cls).__new__(cls, *cargs, **ckwargs)
            temp._init(*iargs, **ikwargs)
            return temp
        
        def Push(result, objs):
            if objs == None:
                return
            if flat and type(objs) == list:
                result.extend(objs)
            else:
                result.append(objs)
        
        def inSync(lod, **kwa):
            
            def dict_image(lod):
                lod.update(k)
                return newInstance([], k, [], lod)
            
            k = {}
            k.update(kwa)
            k.update(kwargs)
            dummy_s = type('dummyClass', (), {'_syncObject':None})()
            if not global_sync:
                dummy_s._syncObject = _Sync_Container(None,
                        k['besideTo']._parentWindow)
                k['syncTo'] = dummy_s
                
            if type(lod) == dict and lod.has_key('tabTitle'):
                k['syncTo']._syncObject.set_title(lod['tabTitle'])
                lod = lod['images']
            if type(lod) == dict:
                return dict_image(lod)
            if type(lod) == list:
                result = []
                for x in lod:
                    if type(x) == dict:
                        result.append(dict_image(x))
                    else:
                        result.append(newInstance([], k, [x], k))
                return result
            if type(lod) == np.ndarray:
                return newInstance([], k, [lod], k)
            print 'left inSync via unknown type:', str(type(lod))
        
        def inWindow(lod):
            result = []
            dummy_w = type('dummyClass', (), {'_parentWindow':None})()
            if not global_sync and not global_window:
                dummy_w._parentWindow = _Multi_Image_Container(None)
            
            if type(lod) == np.ndarray:
                Push(result, inSync(lod, besideTo=dummy_w))
                result[-1]._parentWindow.resize2fit()
                return result
            if type(lod) == dict and lod.has_key('windowTitle'):
                dummy_w._parentWindow.set_title(lod['windowTitle'])
                lod = lod['tabs']
            if type(lod) == dict:
                    Push(result, inSync(lod, besideTo=dummy_w))
                    #result[-1]._parentWindow.resize2fit()
                    return result
            if type(lod) == list:
                for x in lod:
                    Push(result, inSync(x, besideTo=dummy_w))
                unfold(result, True)._parentWindow.resize2fit()
                return result
            else:
                raise TypeError('ndarray or list-/dict-type expected, got: '+
                        str(type(lod)))
                        
        def unfold(obj, return_last_obj=False):
            o = obj
            while type(o) == list and len(o) == 1:
                o = o[0]
            if type(o) == ImageVisualizer:
                return o
            elif return_last_obj:
                while type(o) == list:
                    o = o[-1]
                return o
            else:
                return obj
        
        if len(args) == 0:
            return newInstance(args, kwargs, args, kwargs)
        
        if len(args) > 1:
            if type(args[0]) == np.ndarray and type(args[1]) == str():
                return newInstance(args, kwargs, args, kwargs)
            kwargs.pop('ndArr', None)           # global_imagetitle/-info not
            kwargs.pop('imgTitle', None)        # allowed on multiple images
            kwargs.pop('imgInfo', None)
        
        global_sync = kwargs.has_key('syncTo')
        global_window = kwargs.has_key('besideTo')
        flat = kwargs.pop('flat', False)
        result = []
        for x in args:
            Push(result, inWindow(x))
        
        r = unfold(result)
        _gT.temp = r #TODO remove this testcode
        return r
        
    def __init__(*args, **kwargs):
        pass
    
    def _init(self, ndArr=None, imgTitle=None, imgInfo=None,
            colortable=None, syncTo=None, besideTo=None):
        
        #print 'init with', type(ndArr), type(imgTitle), type(imgInfo), \
        #    type(colortable), type(syncTo), type(besideTo)
        super(ImageVisualizer, self).__init__()
        self._id = _gT._new_image_id()
        self._title = None
        self._info = None
        self._colt = None
        self._colt_mode = 'default'
        self._img = None
        self._c = self._Communicate()
        self._c.update_image.connect(self._update)
        self._c.deleteLater.connect(self._delLater)
        
        self._lbl_image = QtGui.QLabel()
        self._lbl_title = QtGui.QLabel()
        self._sa = QtGui.QScrollArea()
        self._te = QtGui.QTextEdit()
        self._vbl = QtGui.QVBoxLayout()
        
        self._sa.setWidgetResizable(False)
        self._sa.setBackgroundRole(QtGui.QPalette.Dark)
        self._sa.setAlignment(QtCore.Qt.AlignCenter)
        self._sa.setWidget(self._lbl_image)
        self._vbl.addWidget(self._sa)
        self._vbl.setStretch(0, 3)
        self._te.setReadOnly(True)
        
        if syncTo != None:
            self._syncObject = syncTo._syncObject.register(self)
        elif besideTo != None:
            self._syncObject = _Sync_Container(self, besideTo._parentWindow)
        else:
            self._syncObject = _Sync_Container(self)
        self._parentWindow = self._syncObject._parentWindow
        
        if ndArr != None:
            self.image(ndArr, imgTitle, imgInfo, colortable)
        
    def __del__(self):
        if hasattr(self, '_vbl'):
            self._vbl.setParent(None)
            self._te.setParent(None)
            self._lbl_title.setParent(None)
            self._lbl_image.setParent(None)
            self._sa.setParent(None)
            
            #self._ref2self = self
            self._c.deleteLater.emit([self])
        """self._vbl.deleteLater()
        self._te.deleteLater()
        self._lbl_title.deleteLater()
        self._lbl_image.deleteLater()
        self._sa.deleteLater()"""
        
    def _delLater(self, r2s):
        del self._te
        del self._lbl_title
        del self._lbl_image
        del self._sa
        del self._vbl
        print 'deleteLater of Img'+str(self._id)+' executed'
        #del self._ref2self
    
    def _forceShow(self, imgTitle=False, imgInfo=False):
        self._forced = True
        if imgTitle and self._title == None:
            self.set_title('Image'+str(self._id))
        if imgInfo and self._info == None:
            self.set_info('')
        del self._forced
        
    def image(self, ndArr, imgTitle=None, imgInfo=None, colortable=None):
        if imgTitle != None:
            self.set_title(imgTitle)
        if imgInfo != None:
            self.set_info(imgInfo)
        if colortable != None:
            self.set_colortable(colortable)
        
        if self.swapaxis:
            c = np.require(ndArr.T, np.uint32, 'C')
        else:
            c = np.require(ndArr, np.uint32, 'C')
        if self._colt_mode == 'default':
            t = ndArr.dtype
            col = None
            if t == np.bool_ or t == np.bool8:
                col = np.array([((x*256+x)*256+x) for x in [0, 255]],
                    np.uint32)
            elif t == np.uint8:
                col = np.array([((x*256+x)*256+x) for x in range(256)],
                    np.uint32)
            elif t == np.int8:
                col = np.zeros(256, dtype=np.uint32)
                for x in range(100): col[x] = int(255*x/100.)*2**16
                for x in range(100,128): col[x] = 255*2**16 + x*3*2**8
                for x in range(128,184): col[x] = 255 + (170-3*x)*2**8
                for x in range(184,256): col[x] = 255*(1-1/72.*(x-184)) 
            if col != None:
                c.flat[...] = col[c.flat[...]]
        elif self._colt != None:
            c.flat[...] = self._colt[c.flat[...]]
        
        self._img = QtGui.QImage(c.data, c.shape[1], c.shape[0],
                QtGui.QImage.Format_RGB32)
        self._c.update_image.emit()
        
    def _update(self):
        self._lbl_image.setPixmap(QtGui.QPixmap.fromImage(self._img))
        self._lbl_image.resize(self._img.width(), self._img.height())
        
    def add_image(self, ndArr, *args, **kwargs):
        kwargs['besideTo'] = self
        return ImageVisualizer(ndArr, *args, **kwargs)
            
    def add_sync_image(self, ndArr, *args, **kwargs):
        kwargs['syncTo'] = self
        return ImageVisualizer(ndArr, *args, **kwargs)
            
    def set_title(self, imgTitle):
        if self._title == None and imgTitle != None:
            self._vbl.insertWidget(0, self._lbl_title)
            if not hasattr(self, '_forced'):
                self._syncObject.show_imagetitle()
        self._title = imgTitle
        self._lbl_title.setText(self._title)
        
    def set_info(self, imgInfo):
        if self._info == None and imgInfo != None:
            self._vbl.addWidget(self._te)
            if self._title != None:
                self._vbl.setStretch(2, 1)
            else:
                self._vbl.setStretch(1, 1)
            if not hasattr(self, '_forced'):
                self._syncObject.show_info()
        self._info = imgInfo
        self._te.setPlainText(imgInfo)
        
    def set_colortable(self, colortable):
        self._colt = colortable
        self._colt_mode = 'default' if colortable == None else 'manuel'
        
    def set_tab_title(self, title):
        self._syncObject.set_title(title)
        
    def set_window_title(self, title):
        self._parentWindow.set_title(title)
        
    def show(self):
        self._syncObject.show()

class _Sync_Container(object):
    """"""#TODO
    
    class _Communicate(QtCore.QObject):
        deleteLater = QtCore.pyqtSignal()
    
    def __init__(self, imgVis, parentWindow=None):
        self._id = _gT._new_sync_id()
        self._syncList = {}
        self._titleVisible = False
        self._infoVisible = False
        self._c = self._Communicate()
        self._c.deleteLater.connect(self._deleteLater)
        self._hbl = QtGui.QHBoxLayout()
        self._hbl.setMargin(0)
        self._hbl.setSpacing(0)
        if imgVis != None:
            self.register(imgVis)
            if imgVis._title == None:
                title = 'Image'+str(imgVis._id)
            else:
                title = imgVis._title
        else:
            title = 'Sync'+str(self._id+1)
        self._title = title
        if parentWindow == None:
            self._parentWindow = _Multi_Image_Container(self)
        else:
            self._parentWindow = parentWindow.register(self)
            
    def __del__(self):
        if hasattr(self, '_hbl'):
            self._hbl.setParent(None)
            
            #self._ref2self = self
            #_gT.__dict__['Sync'+str(self._id)] = self
            print 'Sync'+str(self._id)+' emitting deleteLater.'
            self._c.deleteLater.emit()
        
    def _deleteLater(self):
        del self._set_title_callback
        del self._show_tab_callback
        del self._hbl
        print 'Sync'+str(self._id)+' deleteLater executed.'
        del _gT.__dict__['Sync'+str(self._id)]
        #del self._ref2self
        
    def register(self, imgVis):
        r = weakref.ref(imgVis, self._deregister)
        self._syncList[id(r)] = {'ref':r, 'id':imgVis._id,
                'hsb':True, 'vsb':True}
        self._hbl.addLayout(imgVis._vbl)
        self._titleVisible |= imgVis._title != None
        if self._titleVisible:
            self.show_imagetitle()
        self._infoVisible |= imgVis._info != None
        if self._infoVisible:
            self.show_info()
        imgVis._sa.horizontalScrollBar().valueChanged.connect(
                _ext_Callback(self._move_scrollbar, imgVis._id, 'hsb'))
        imgVis._sa.verticalScrollBar().valueChanged.connect(
                _ext_Callback(self._move_scrollbar, imgVis._id, 'vsb'))
        return self
        
    def _deregister(self, ref):
        del self._syncList[id(ref)]
        
    def show_imagetitle(self):
        self._titleVisible = True
        for x in self._syncList.itervalues():
            r = x['ref']()
            if r != None:
                r._forceShow(True, False)
            
    def show_info(self):
        self._infoVisible = True
        for x in self._syncList.itervalues():
            r = x['ref']()
            if r != None:
                r._forceShow(False, True)
            
    def _move_scrollbar(self, value, imgID, sbid):
        x = {'id':-1}
        sync = False
        for i in self._syncList.itervalues():
            if i['id'] == imgID:
                if i[sbid] == True:
                    x = i
                    if sbid == 'hsb':
                        sb = i['ref']()._sa.horizontalScrollBar()
                    elif sbid == 'vsb':
                        sb = i['ref']()._sa.verticalScrollBar()
                    mini = sb.minimum()
                    maxi = sb.maximum()
                    diff = float(maxi-mini)
                    value = 0 if diff == 0 else (value-mini)/diff
                    sync = True
                else:
                    i[sbid] = True
                    break
        
        if sync:
            for i in self._syncList.itervalues():
                r = i['ref']()
                if r != None and i['id'] != x['id']:
                    i[sbid] = False
                    if sbid == 'hsb':
                        sb = r._sa.horizontalScrollBar()
                    elif sbid == 'vsb':
                        sb = r._sa.verticalScrollBar()
                    mini = sb.minimum()
                    maxi = sb.maximum()
                    new = (maxi - mini)*value + mini
                    sb.setValue(new)
                        
    def set_title(self, tabtitle):
        self._title = tabtitle
        self._set_title_callback(tabtitle)
        
    def show(self):
        self._show_tab_callback()
        self._parentWindow.show()
        

class _Multi_Image_Container(object):
    """"""#TODO
    
    class _Communicate(QtCore.QObject):
        deleteLater = QtCore.pyqtSignal(list)
    
    def __init__(self, sync_obj):
        self._id = _gT._new_win_id()
        self._tabcount = 0
        self._tablist = {}
        self._c = self._Communicate()
        self._c.deleteLater.connect(self._deleteLater)
        self._tw = QtGui.QTabWidget()
        self._hbl = QtGui.QHBoxLayout()
        self._hbl.setMargin(0)
        self._wi = QtGui.QWidget()
        if sync_obj == None or len(sync_obj._syncList) == 0 or \
                sync_obj._syncList.values()[0]['ref']()._title == None:
            title = 'ImageWindow'+str(self._id+1)
        else:
            title = sync_obj._syncList.values()[0]['ref']()._title
        self._wi.setWindowTitle(title)
        self._wi.setLayout(self._hbl)
        if sync_obj != None:
            self.register(sync_obj)
        self._wi.show()
        
    def __del__(self):
        if hasattr(self, '_tw'):
            self._tw.setParent(None)
            self._hbl.setParent(None)
            
            #self._ref2self = self
            self._c.deleteLater.emit([self])
        
    def _deleteLater(self, r2s):
        del self._tw
        del self._hbl
        del self._wi
        #del self._ref2self
        
    def register(self, sync_obj):
        r = weakref.ref(sync_obj, self._deregister)
        entry = {'ref':r, 'tab':None, 'id':sync_obj._id}
        self._tablist[id(r)] = entry
        sync_obj._set_title_callback = _ext_Callback(self._set_tab_title,
                self._tablist[id(r)])
        sync_obj._show_tab_callback = _ext_Callback(self._show_tab,
                self._tablist[id(r)])
        self._tabcount += 1
        if self._tabcount == 1:
            self._hbl.addLayout(sync_obj._hbl)
        else:
            if self._tabcount == 2:
                self._hbl.setMargin(4)
                self._hbl.addWidget(self._tw)
                for x in self._tablist.itervalues():
                    obj = x['ref']()
                    obj._hbl.setParent(None)
                    x['tab'] = QtGui.QWidget()
                    x['tab'].setLayout(obj._hbl)
                    self._tw.addTab(x['tab'], obj._title)
            else:
                entry['tab'] = QtGui.QWidget()
                entry['tab'].setLayout(sync_obj._hbl)
                self._tw.addTab(entry['tab'], sync_obj._title)
        return self
        
    def _deregister(self, ref):
        if self._tabcount > 1:
            self._tw.removeTab(self._tw.indexOf(self._tablist[id(ref)]['tab']))
        del self._tablist[id(ref)]
        self._tabcount -= 1
        
    def set_title(self, title):
        self._wi.setWindowTitle(title)
        
    def _set_tab_title(self, tabtitle, reg_entry):
        tab = reg_entry['tab']
        if tab != None:
            self._tw.setTabText(self._tw.indexOf(tab), tabtitle)
        
    def show(self):
        self._wi.show()
            
    def _show_tab(self, reg_entry):
        tab = reg_entry['tab']
        if tab != None:
            self._tw.setCurrentWidget(tab)
            
    def resize2fit(self):
        if self._tabcount > 1:
            s = self._tw.sizeHint()
        else:
            obj = self._tablist.values()[0]['ref']()
            if obj == None:
                s = QtGui.QSize(240,180)
            else:
                obj = obj._syncList.values()[0]['ref']()
                if obj == None:
                    s = QtGui.QSize(240,180)
                else:
                    s = obj._lbl_image.sizeHint()
        w = s.width() + 8
        h = s.height() + 8
        self._wi.resize(w, h)


class _guiThread(Thread):
    """
    Provides easy access to #TODO Windows through managing only one QApplication
    in an seperated Thread.
    """
    _WinCount = 0
    _ImageCount = 0
    _SyncCount = 0
    
    def __init__(self):
        super (_guiThread, self).__init__()
        self._app = QtGui.QApplication(sys.argv)
        self._app.setQuitOnLastWindowClosed(False)
        self.start()
        
    def run(self):
        #self._app.exec_()
        pass
        
    def _new_image_id(self):
        x = self._ImageCount
        self._ImageCount += 1
        return x
        
    def _new_sync_id(self):
        x = self._SyncCount
        self._SyncCount += 1
        return x
        
    def _new_win_id(self):
        x = self._WinCount
        self._WinCount += 1
        return x
        
        
def colorize(a, color):
    """Colorize a with color. MinValue of a becomes black and MaxValue
    becomes the color defined by the 'color'-RGB-Vector. Other Values
    are evenly, linearly distributed between those."""
    b = np.require(a, np.uint32, 'C')
    mini, maxi = a.min(), a.max()
    d = float(maxi-mini)
    if d == 0:
        return b
    colt = np.array([((int(color[0]*x/255.)*256 + int(color[1]*x/255.))*256 + \
            int(color[2]*x/255.)) for x in range(256)], np.uint32)
    b.flat[...] = colt[np.require((b.flat[...]-mini)*255/d, np.uint32)]
    return b



ImVis = ImageVisualizer # short Name for external use
_gT = _guiThread()
_gT.temp = [] #TODO remove this testcode

if __name__ == '__main__':
    a = np.random.random(640*480)*255
    a = a.reshape((640,480))
    win = ImageVisualizer(np.require(a, np.uint8, 'C'), imgTitle='Randomvalues')
    
    def updateData():
        global a
        a=np.roll(a,-100)
        win.image(a)
    
    from threading import Timer
    t = Timer(1.0, updateData)
    t.start()
    import time
    time.sleep(5)








