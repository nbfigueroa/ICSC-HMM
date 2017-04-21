function draw3DArrowN(pos,dir,w,col)
  plot3([pos(1) pos(1)+dir(1)], [pos(2) pos(2)+dir(2)], [pos(3) pos(3)+dir(3)], '-','linewidth',w,'color',col);
  sz = 5E-3;
  pos = pos+dir;
  dir = dir/norm(dir);
  prp = [dir(2); -dir(1); dir(3)];
  dir = dir*sz;
  prp = prp*sz;
  msh = [pos pos-dir-prp/2 pos-dir+prp/2 pos];
  patch(msh(1,:),msh(2,:),msh(3,:),col,'edgecolor',col);
end