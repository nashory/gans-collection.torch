-- this script will run server for display.
-- last modified : 2017.09.01, nashory


local opts = require '__5_ebgan.script.opts'
local opt = opts.parse(arg)

if opt.display then
	os.execute(string.format('th -ldisplay.start %d %s', opt.display_server_port, opt.display_server_ip))
end


  
