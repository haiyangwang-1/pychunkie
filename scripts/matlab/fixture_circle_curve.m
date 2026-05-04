function [r, d, d2] = fixture_circle_curve(t, radius)
%FIXTURE_CIRCLE_CURVE Circle curve with configurable radius.

r = radius * [cos(t(:).'); sin(t(:).')];
d = radius * [-sin(t(:).'); cos(t(:).')];
d2 = radius * [-cos(t(:).'); -sin(t(:).')];
end
