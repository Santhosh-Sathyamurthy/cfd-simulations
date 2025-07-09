# MIT License
# Copyright (c) 2025 Santhosh S
# See LICENSE file for full license text.

using Printf

# Simple video generation parameters
const BASE_DIR = "cylinder_flow_output"
const VIDEO_DIR = "cylinder_flow_videos"
const RE_VALUES = [0.1, 1.0, 10.0, 40.0]
const FRAMERATE = 10  # fps
const GIF_FRAMERATE = 10  

function setup_video_directories()
    """Setup output directories for videos"""
    isdir(VIDEO_DIR) || mkdir(VIDEO_DIR)
    println("Created video directory: $VIDEO_DIR")
end

function check_ffmpeg()
    """Check if FFmpeg is available"""
    try
        run(`ffmpeg -version`)
        println("✓ FFmpeg found")
        return true
    catch
        println("✗ FFmpeg not found. Please install FFmpeg first.")
        return false
    end
end

function create_video_from_pngs(input_dir, output_file, pattern)
    """Create video/GIF from PNG sequence"""
    
    # Check if input directory exists
    if !isdir(input_dir)
        println("  ✗ Directory not found: $input_dir")
        return false
    end
    
    # Create input pattern for ffmpeg (assuming files are named like pattern_0000.png)
    input_pattern = joinpath(input_dir, pattern)
    
    # Check if at least the first file exists
    first_file = joinpath(input_dir, replace(pattern, "%04d" => "0000"))
    if !isfile(first_file)
        println("  ✗ No files found matching pattern: $pattern")
        return false
    end
    
    try
        if endswith(output_file, ".gif")
            # For GIF creation with palette optimization
            temp_palette = joinpath(input_dir, "temp_palette.png")
            
            # Generate palette
            cmd1 = `ffmpeg -y -framerate $GIF_FRAMERATE -i $input_pattern -vf "fps=$GIF_FRAMERATE,scale=640:480:flags=lanczos,palettegen" $temp_palette`
            run(cmd1)
            
            # Create GIF using palette
            cmd2 = `ffmpeg -y -framerate $GIF_FRAMERATE -i $input_pattern -i $temp_palette -lavfi "fps=$GIF_FRAMERATE,scale=640:480:flags=lanczos[x];[x][1:v]paletteuse" $output_file`
            run(cmd2)
            
            # Clean up palette
            rm(temp_palette)
        else
            # For MP4 creation
            cmd = `ffmpeg -y -framerate $FRAMERATE -i $input_pattern -c:v libx264 -pix_fmt yuv420p -vf "scale=640:480" $output_file`
            run(cmd)
        end
        
        println("  ✓ Created: $(basename(output_file))")
        return true
        
    catch e
        println("  ✗ Failed: $e")
        return false
    end
end

function count_files(input_dir, base_pattern)
    """Count how many files match the pattern"""
    if !isdir(input_dir)
        return 0
    end
    
    count = 0
    for i in 0:999  # Check up to 1000 files
        # Create the filename using @sprintf with literal format string
        filename = @sprintf("%s_%04d.png", base_pattern, i)
        filepath = joinpath(input_dir, filename)
        if isfile(filepath)
            count += 1
        else
            # Check if we've reached the end of consecutive files
            if i > 0
                break
            end
        end
    end
    return count
end

function find_file_range(input_dir, base_pattern)
    """Find the actual range of files that exist"""
    if !isdir(input_dir)
        return 0, 0
    end
    
    min_idx = -1
    max_idx = -1
    
    # Check from 0 to 999 (should be enough for most cases)
    for i in 0:999
        filename = @sprintf("%s_%04d.png", base_pattern, i)
        filepath = joinpath(input_dir, filename)
        if isfile(filepath)
            if min_idx == -1
                min_idx = i
            end
            max_idx = i
        end
    end
    
    return min_idx, max_idx
end

function main()
    """Main function - create videos for each Reynolds number"""
    
    println("="^50)
    println("CYLINDER FLOW VIDEO GENERATOR")
    println("="^50)
    
    # Check FFmpeg
    if !check_ffmpeg()
        return
    end
    
    # Setup directories
    setup_video_directories()
    
    # Check base directory
    if !isdir(BASE_DIR)
        println("✗ Simulation output directory not found: $BASE_DIR")
        return
    end
    
    println("Processing Reynolds numbers: $(RE_VALUES)")
    println("Video framerate: $(FRAMERATE) fps")
    println("GIF framerate: $(GIF_FRAMERATE) fps")
    println()
    
    successful = 0
    total = 0
    
    # Process each Reynolds number
    for Re in RE_VALUES
        println("Processing Re = $Re")
        println("-"^30)
        
        input_dir = joinpath(BASE_DIR, "Re_$(Re)")
        
        if !isdir(input_dir)
            println("  ✗ Directory not found: $input_dir")
            continue
        end
        
        # Count available files and find ranges
        vort_count = count_files(input_dir, "vorticity")
        vel_count = count_files(input_dir, "velocity")
        
        vort_min, vort_max = find_file_range(input_dir, "vorticity")
        vel_min, vel_max = find_file_range(input_dir, "velocity")
        
        println("  Available files:")
        println("    Vorticity: $vort_count files ($(vort_min)-$(vort_max))")
        println("    Velocity: $vel_count files ($(vel_min)-$(vel_max))")
        
        # Create vorticity videos
        if vort_count > 0
            # MP4
            vort_mp4 = joinpath(VIDEO_DIR, "vorticity_Re_$(Re).mp4")
            total += 1
            if create_video_from_pngs(input_dir, vort_mp4, "vorticity_%04d.png")
                successful += 1
            end
            
            # GIF
            vort_gif = joinpath(VIDEO_DIR, "vorticity_Re_$(Re).gif")
            total += 1
            if create_video_from_pngs(input_dir, vort_gif, "vorticity_%04d.png")
                successful += 1
            end
        else
            println("  ✗ No vorticity files found")
        end
        
        # Create velocity videos
        if vel_count > 0
            # MP4
            vel_mp4 = joinpath(VIDEO_DIR, "velocity_Re_$(Re).mp4")
            total += 1
            if create_video_from_pngs(input_dir, vel_mp4, "velocity_%04d.png")
                successful += 1
            end
            
            # GIF
            vel_gif = joinpath(VIDEO_DIR, "velocity_Re_$(Re).gif")
            total += 1
            if create_video_from_pngs(input_dir, vel_gif, "velocity_%04d.png")
                successful += 1
            end
        else
            println("  ✗ No velocity files found")
        end
        
        println()
    end
    
    # Summary
    println("="^50)
    println("RESULTS")
    println("="^50)
    println("Success: $successful/$total videos/GIFs created")
    println("Output directory: $VIDEO_DIR")
    println()
    
    # List created files with actual sizes
    if isdir(VIDEO_DIR)
        files = filter(f -> endswith(f, ".mp4") || endswith(f, ".gif"), readdir(VIDEO_DIR))
        if !isempty(files)
            println("Created files:")
            for file in sort(files)
                file_path = joinpath(VIDEO_DIR, file)
                if isfile(file_path)
                    size_mb = round(filesize(file_path) / 1024^2, digits=2)
                    println("  - $file ($(size_mb) MB)")
                end
            end
        else
            println("No videos or GIFs were created.")
        end
    end
end

# Run the script
main()