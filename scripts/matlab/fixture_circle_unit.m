function [r, d, d2] = fixture_circle_unit(t)
%FIXTURE_CIRCLE_UNIT Unit circle curve.

r = [cos(t(:).'); sin(t(:).')];
d = [-sin(t(:).'); cos(t(:).')];
d2 = [-cos(t(:).'); -sin(t(:).')];
end
