function fields = fixture_pack_chunker(chnkr)
%FIXTURE_PACK_CHUNKER Pack stable chunker fields into a saved struct.

fields = [];
fields.r = chnkr.r;
fields.d = chnkr.d;
fields.d2 = chnkr.d2;
fields.n = chnkr.n;
fields.wts = chnkr.wts;
fields.adj = chnkr.adj;
fields.tstor = chnkr.tstor;
fields.wstor = chnkr.wstor;
fields.k = chnkr.k;
fields.nch = chnkr.nch;
fields.dim = chnkr.dim;
fields.area = area(chnkr);
fields.chunklen = chunklen(chnkr);
end
