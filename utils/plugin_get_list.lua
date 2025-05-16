-- Script to list all installed REAPER plugins and save to CSV
-- Output format: Index,Name,Identifier,Plugin_Type,Filename,Path,Is_Instrument

local file = io.open("reaper_plugins.csv", "w")

-- Write header
file:write("Index,Name,Identifier,Plugin_Type,Filename,Path,Is_Instrument\n")

local idx = 0
local count = 0
local retval, name, identifier

-- Enumerate all FX
repeat
    retval, name, identifier = reaper.EnumInstalledFX(idx)
    if retval then
        -- Determine plugin type
        local plugin_type = "unknown"
        local is_instrument = "no"
local filename = ""
local path = ""

-- Extract filename and path from identifier
if identifier:find("<") then
    -- Get the part before the < character
    path = identifier:match("^(.+)<")
    if path then
        -- Extract filename from the end of the path
        filename = path:match("([^\\/]+)$")
        -- Remove filename from path to get directory
        path = path:gsub(filename .. "$", "")
    end
end
--     -- Determine plugin type based on name and identifier
        if name:find("VST3i:") or name:find("VSTi:") or identifier:find("VSTi") then
            is_instrument = "yes"
        end
        
        if name:find("VST3:") or name:find("VST3i:") then
            plugin_type = "vst3"
        elseif name:find("VST:") or name:find("VSTi:") or identifier:find(".dll") or identifier:find(".vst") or identifier:find(".so") or identifier:find(".dylib") then
            plugin_type = "vst2"
        elseif name:find("Component:") or identifier:find(".component") then
            plugin_type = "component"
        elseif name:find("JS:") then
            plugin_type = "js"
        elseif name:find("AU:") then
            plugin_type = "au"
        elseif name:find("CLAP:") or identifier:find(".clap") then
            plugin_type = "clap"
        elseif name:find("LV2:") or identifier:find(".lv2") then
            plugin_type = "lv2"
        end
        
        -- Write to CSV
        file:write(count .. "," .. name .. "," .. identifier .. "," .. 
                plugin_type .. "," .. filename .. "," .. 
                path .. "," .. is_instrument .. "\n")
        count = count + 1
    end
    idx = idx + 1
until not retval

file:close()