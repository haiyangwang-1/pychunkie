function [r, d, d2] = fixture_wobbly_curve(t)
%FIXTURE_WOBBLY_CURVE Smooth noncircular curve for chunker parity.

rad = 1.0 + 0.18*cos(3*t(:).');
drad = -0.54*sin(3*t(:).');
d2rad = -1.62*cos(3*t(:).');
ct = cos(t(:).');
st = sin(t(:).');
r = [rad.*ct; rad.*st];
d = [drad.*ct - rad.*st; drad.*st + rad.*ct];
d2 = [d2rad.*ct - 2*drad.*st - rad.*ct; ...
      d2rad.*st + 2*drad.*ct - rad.*st];
end
