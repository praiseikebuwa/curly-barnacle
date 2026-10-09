from __future__ import annotations
import json,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class TrainingReport:
    epochs:int; train_loss:list[float]; validation_loss:list[float]; validation_accuracy:list[float]; checkpoint:str; seed:int; duration_seconds:float; architecture:str='deep_numpy_cnn'

class TinyVisionClassifier:
    """Compatibility wrapper for the deeper CNN model."""
    def __new__(cls,*args,**kwargs): return DeepVisionCNN(*args,**kwargs)

class DeepVisionCNN:
    """Two-convolution, max-pooling classifier with random initialization and NumPy backpropagation."""
    def __init__(self,input_dim=256,classes=6,seed=7):
        side=int(np.sqrt(input_dim)); assert side*side==input_dim
        self.side,self.input_dim,self.classes,self.seed=side,input_dim,classes,seed; rng=np.random.default_rng(seed)
        self.w1=(rng.standard_normal((3,3,1,4))*np.sqrt(2/9)).astype(np.float32); self.b1=np.zeros(4,np.float32)
        self.w2=(rng.standard_normal((3,3,4,8))*np.sqrt(2/36)).astype(np.float32); self.b2=np.zeros(8,np.float32)
        pooled_side=(side-4)//2
        flattened=pooled_side*pooled_side*8
        self.w3=(rng.standard_normal((flattened,classes))*np.sqrt(2/flattened)).astype(np.float32); self.b3=np.zeros(classes,np.float32)
    @staticmethod
    def _conv(x,w,b):
        n,h,ww,c=x.shape; kh,kw,_,oc=w.shape; out=np.empty((n,h-kh+1,ww-kw+1,oc),np.float32)
        for i in range(out.shape[1]):
            for j in range(out.shape[2]): out[:,i,j,:]=np.einsum('nxyc,xyco->no',x[:,i:i+kh,j:j+kw,:],w)+b
        return out
    @staticmethod
    def _conv_backward(x,w,dout):
        dw=np.zeros_like(w); dx=np.zeros_like(x)
        for i in range(dout.shape[1]):
            for j in range(dout.shape[2]):
                patch=x[:,i:i+3,j:j+3,:]; dw+=np.einsum('nxyc,no->xyco',patch,dout[:,i,j,:]); dx[:,i:i+3,j:j+3,:]+=np.einsum('no,xyco->nxyc',dout[:,i,j,:],w)
        return dx,dw,dout.sum((0,1,2))
    def _forward(self,x,cache=False):
        x=x.reshape((-1,self.side,self.side,1)).astype(np.float32); z1=self._conv(x,self.w1,self.b1); a1=np.maximum(z1,0); z2=self._conv(a1,self.w2,self.b2); a2=np.maximum(z2,0); pooled=a2.reshape(a2.shape[0],a2.shape[1]//2,2,a2.shape[2]//2,2,a2.shape[3]).max((2,4)); flat=pooled.reshape(len(x),-1); logits=flat@self.w3+self.b3; logits-=logits.max(1,keepdims=True); probs=np.exp(logits); probs/=probs.sum(1,keepdims=True); return (probs,(x,z1,a1,z2,a2,pooled,flat)) if cache else probs
    def predict(self,x): return self._forward(x).argmax(1)
    def loss_accuracy(self,x,y):
        p=self._forward(x); return float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-8)).mean()),float((p.argmax(1)==y).mean())
    def fit(self,x_train,y_train,x_val,y_val,epochs=25,learning_rate=.1,checkpoint='checkpoints/tiny_vision.npz'):
        x_train=np.asarray(x_train,np.float32); x_val=np.asarray(x_val,np.float32); y_train=np.asarray(y_train,np.int64); y_val=np.asarray(y_val,np.int64); start=time.time(); tl=[];vl=[];va=[]
        for _ in range(epochs):
            p,(x,z1,a1,z2,a2,pooled,flat)=self._forward(x_train,True); one=np.zeros_like(p); one[np.arange(len(y_train)),y_train]=1; dz=(p-one)/len(y_train); dw3=flat.T@dz; db3=dz.sum(0); dflat=dz@self.w3.T; dpool=dflat.reshape(pooled.shape); da2=np.zeros_like(a2); da2[:,::2,::2,:]+=dpool; da2[:,1::2,::2,:]+=dpool; da2[:,::2,1::2,:]+=dpool; da2[:,1::2,1::2,:]+=dpool; dz2=da2*(z2>0); da1,dw2,db2=self._conv_backward(a1,self.w2,dz2); dz1=da1*(z1>0); _,dw1,db1=self._conv_backward(x,self.w1,dz1); self.w3-=learning_rate*dw3; self.b3-=learning_rate*db3; self.w2-=learning_rate*dw2; self.b2-=learning_rate*db2; self.w1-=learning_rate*dw1; self.b1-=learning_rate*db1; l,_=self.loss_accuracy(x_train,y_train); l2,a=self.loss_accuracy(x_val,y_val); tl.append(l);vl.append(l2);va.append(a)
        report=TrainingReport(epochs,tl,vl,va,str(checkpoint),self.seed,time.time()-start); self.save(checkpoint); return report
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,w1=self.w1,b1=self.b1,w2=self.w2,b2=self.b2,w3=self.w3,b3=self.b3,side=self.side,input_dim=self.input_dim,classes=self.classes,seed=self.seed)
    @classmethod
    def load(cls,path):
        d=np.load(path); m=cls(int(d['input_dim']),int(d['classes']),int(d['seed']));m.w1,m.b1,m.w2,m.b2,m.w3,m.b3=d['w1'],d['b1'],d['w2'],d['b2'],d['w3'],d['b3'];return m

def save_report(report,path): Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(report.__dict__,indent=2),encoding='utf-8')
