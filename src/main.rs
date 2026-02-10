use std::ffi::c_int;

use cudarc::driver::result as cuda;
use cudarc::driver::sys::{self, CUdevice_attribute::*};

/// Maps SM version (major, minor) to CUDA cores per SM.
fn sm_to_cores(major: i32, minor: i32) -> Option<i32> {
    match (major, minor) {
        // Kepler
        (3, 0) | (3, 2) | (3, 5) | (3, 7) => Some(192),
        // Maxwell
        (5, 0) | (5, 2) | (5, 3) => Some(128),
        // Pascal
        (6, 0) => Some(64),
        (6, 1) | (6, 2) => Some(128),
        // Volta / Turing
        (7, 0) | (7, 2) | (7, 5) => Some(64),
        // Ampere
        (8, 0) => Some(64),
        (8, 6) | (8, 7) | (8, 9) => Some(128),
        // Hopper / Ada Lovelace
        (9, 0) => Some(128),
        // Blackwell and beyond
        (10, 0) | (10, 1) | (10, 3) => Some(128),
        (11, 0) => Some(128),
        (12, 0) | (12, 1) => Some(128),
        _ => None,
    }
}

/// Queries a single integer device attribute, returning 0 on failure.
fn attr(dev: sys::CUdevice, a: sys::CUdevice_attribute) -> i32 {
    unsafe { cuda::device::get_attribute(dev, a).unwrap_or(0) }
}

fn driver_version() -> Result<c_int, cudarc::driver::DriverError> {
    let mut version: c_int = 0;
    let result = unsafe { sys::cuDriverGetVersion(&mut version) };
    if result != sys::CUresult::CUDA_SUCCESS {
        // Return 0 wrapped in Ok rather than trying to convert the raw CUresult
        return Ok(0);
    }
    Ok(version)
}

fn can_access_peer(dev: sys::CUdevice, peer: sys::CUdevice) -> bool {
    let mut can_access: c_int = 0;
    let result = unsafe { sys::cuDeviceCanAccessPeer(&mut can_access, dev, peer) };
    result == sys::CUresult::CUDA_SUCCESS && can_access != 0
}

fn compute_mode_str(mode: i32) -> &'static str {
    match mode {
        0 => "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
        1 => {
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)"
        }
        2 => "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
        3 => {
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)"
        }
        _ => "Unknown",
    }
}

fn main() {
    // Initialize the CUDA driver API
    if let Err(e) = cuda::init() {
        eprintln!("Failed to initialize CUDA driver: {:?}", e);
        std::process::exit(1);
    }

    let driver_ver = driver_version().unwrap_or(0);

    let dev_count = match cuda::device::get_count() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to get device count: {:?}", e);
            std::process::exit(1);
        }
    };

    if dev_count == 0 {
        println!("There are no available device(s) that support CUDA");
        std::process::exit(0);
    }

    println!("Detected {} CUDA Capable device(s)\n", dev_count);

    let mut devices = Vec::new();

    for i in 0..dev_count {
        let dev = match cuda::device::get(i) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Failed to get device {}: {:?}", i, e);
                continue;
            }
        };
        devices.push(dev);

        let name = cuda::device::get_name(dev).unwrap_or_else(|_| "Unknown".to_string());
        let total_mem = unsafe { cuda::device::total_mem(dev).unwrap_or(0) };

        let major = attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        let minor = attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);

        println!("Device {}: \"{}\"", i, name);
        println!(
            "  CUDA Driver Version:                           {}.{}",
            driver_ver / 1000,
            (driver_ver % 1000) / 10
        );
        println!(
            "  CUDA Capability Major/Minor version number:    {}.{}",
            major, minor
        );

        let total_mem_mb = total_mem as f64 / (1024.0 * 1024.0);
        println!(
            "  Total amount of global memory:                 {:.0} MBytes ({} bytes)",
            total_mem_mb, total_mem
        );

        let mp_count = attr(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        let cores_per_mp = sm_to_cores(major, minor);

        match cores_per_mp {
            Some(cores) => println!(
                "  ({:03}) Multiprocessors, ({:03}) CUDA Cores/MP:    {} CUDA Cores",
                mp_count,
                cores,
                mp_count * cores
            ),
            None => println!(
                "  ({:03}) Multiprocessors (unknown CUDA Cores/MP for SM {}.{})",
                mp_count, major, minor
            ),
        }

        let clock_rate = attr(dev, CU_DEVICE_ATTRIBUTE_CLOCK_RATE); // in kHz
        println!(
            "  GPU Max Clock rate:                            {:.0} MHz ({:.2} GHz)",
            clock_rate as f64 / 1000.0,
            clock_rate as f64 / 1_000_000.0
        );

        let mem_clock = attr(dev, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE); // in kHz
        println!(
            "  Memory Clock rate:                             {:.0} Mhz",
            mem_clock as f64 / 1000.0
        );

        let mem_bus_width = attr(dev, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
        println!(
            "  Memory Bus Width:                              {}-bit",
            mem_bus_width
        );

        let l2_cache = attr(dev, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE);
        if l2_cache > 0 {
            println!(
                "  L2 Cache Size:                                 {} bytes",
                l2_cache
            );
        }

        let max_tex1d = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH);
        let max_tex2d_w = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH);
        let max_tex2d_h = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT);
        let max_tex3d_w = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH);
        let max_tex3d_h = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT);
        let max_tex3d_d = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH);
        println!(
            "  Maximum Texture Dimension Size (x,y,z)         1D=({}) 2D=({}, {}) 3D=({}, {}, {})",
            max_tex1d, max_tex2d_w, max_tex2d_h, max_tex3d_w, max_tex3d_h, max_tex3d_d
        );

        let max_tex1d_layered_w = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH);
        let max_tex1d_layered_l = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS);
        let max_tex2d_layered_w = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH);
        let max_tex2d_layered_h = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT);
        let max_tex2d_layered_l = attr(dev, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS);
        println!(
            "  Maximum Layered 1D Texture Size, (num) layers  1D=({}) {} layers",
            max_tex1d_layered_w, max_tex1d_layered_l
        );
        println!(
            "  Maximum Layered 2D Texture Size, (num) layers  2D=({}, {}) {} layers",
            max_tex2d_layered_w, max_tex2d_layered_h, max_tex2d_layered_l
        );

        let const_mem = attr(dev, CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
        println!(
            "  Total amount of constant memory:               {} bytes",
            const_mem
        );

        let shared_mem = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        println!(
            "  Total amount of shared memory per block:       {} bytes",
            shared_mem
        );

        let shared_mem_mp = attr(
            dev,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
        );
        println!(
            "  Total shared memory per multiprocessor:        {} bytes",
            shared_mem_mp
        );

        let regs = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
        println!("  Total number of registers available per block: {}", regs);

        let warp_size = attr(dev, CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        println!(
            "  Warp size:                                     {}",
            warp_size
        );

        let max_threads_mp = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        println!(
            "  Maximum number of threads per multiprocessor:  {}",
            max_threads_mp
        );

        let max_threads_block = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        println!(
            "  Maximum number of threads per block:           {}",
            max_threads_block
        );

        let max_dim_x = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
        let max_dim_y = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
        let max_dim_z = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
        println!(
            "  Max dimension size of a thread block (x,y,z):  ({}, {}, {})",
            max_dim_x, max_dim_y, max_dim_z
        );

        let max_grid_x = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
        let max_grid_y = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
        let max_grid_z = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
        println!(
            "  Max dimension size of a grid size    (x,y,z):  ({}, {}, {})",
            max_grid_x, max_grid_y, max_grid_z
        );

        let max_pitch = attr(dev, CU_DEVICE_ATTRIBUTE_MAX_PITCH);
        println!(
            "  Maximum memory pitch:                          {} bytes",
            max_pitch
        );

        let tex_align = attr(dev, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
        println!(
            "  Texture alignment:                             {} bytes",
            tex_align
        );

        let gpu_overlap = attr(dev, CU_DEVICE_ATTRIBUTE_GPU_OVERLAP);
        let async_engines = attr(dev, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT);
        println!(
            "  Concurrent copy and kernel execution:          {} with {} copy engine(s)",
            if gpu_overlap != 0 { "Yes" } else { "No" },
            async_engines
        );

        let kernel_timeout = attr(dev, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT);
        println!(
            "  Run time limit on kernels:                     {}",
            if kernel_timeout != 0 { "Yes" } else { "No" }
        );

        let integrated = attr(dev, CU_DEVICE_ATTRIBUTE_INTEGRATED);
        println!(
            "  Integrated GPU sharing Host Memory:            {}",
            if integrated != 0 { "Yes" } else { "No" }
        );

        let can_map = attr(dev, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
        println!(
            "  Support host page-locked memory mapping:       {}",
            if can_map != 0 { "Yes" } else { "No" }
        );

        let surface_align = attr(dev, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT);
        println!(
            "  Alignment requirement for Surfaces:            {}",
            if surface_align != 0 { "Yes" } else { "No" }
        );

        let ecc = attr(dev, CU_DEVICE_ATTRIBUTE_ECC_ENABLED);
        println!(
            "  Device has ECC support:                        {}",
            if ecc != 0 { "Enabled" } else { "Disabled" }
        );

        #[cfg(target_os = "windows")]
        {
            let tcc = attr(dev, CU_DEVICE_ATTRIBUTE_TCC_DRIVER);
            println!(
                "  CUDA Device Driver Mode (TCC or WDDM):        {}",
                if tcc != 0 {
                    "TCC (Tesla Compute Cluster Driver)"
                } else {
                    "WDDM (Windows Display Driver Model)"
                }
            );
        }

        let unified = attr(dev, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
        println!(
            "  Device supports Unified Addressing (UVA):      {}",
            if unified != 0 { "Yes" } else { "No" }
        );

        let managed = attr(dev, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY);
        println!(
            "  Device supports Managed Memory:                {}",
            if managed != 0 { "Yes" } else { "No" }
        );

        let preemption = attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED);
        println!(
            "  Device supports Compute Preemption:            {}",
            if preemption != 0 { "Yes" } else { "No" }
        );

        let coop_launch = attr(dev, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
        println!(
            "  Supports Cooperative Kernel Launch:            {}",
            if coop_launch != 0 { "Yes" } else { "No" }
        );

        let coop_multi = attr(dev, CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH);
        println!(
            "  Supports MultiDevice Co-op Kernel Launch:      {}",
            if coop_multi != 0 { "Yes" } else { "No" }
        );

        let pci_domain = attr(dev, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
        let pci_bus = attr(dev, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
        let pci_device = attr(dev, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
        println!(
            "  Device PCI Domain ID / Bus ID / location ID:   {} / {} / {}",
            pci_domain, pci_bus, pci_device
        );

        let compute_mode = attr(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
        println!("  Compute Mode:",);
        println!("     < {} >", compute_mode_str(compute_mode));

        println!();
    }

    // Peer-to-peer access matrix for multi-GPU systems
    if devices.len() > 1 {
        println!("deviceQuery, Pair-to-Pair GPU Bandwidth Matrix (in GB/s)");
        print!("   D\\D");
        for j in 0..devices.len() {
            print!("{:>6}", j);
        }
        println!();

        for (i, &dev) in devices.iter().enumerate() {
            print!("   {:>3}", i);
            for (j, &peer) in devices.iter().enumerate() {
                if i == j {
                    print!("   Yes");
                } else {
                    let access = can_access_peer(dev, peer);
                    print!("   {}", if access { "Yes" } else { "No" });
                }
            }
            println!();
        }
        println!();
    }

    println!(
        "deviceQuery, CUDA Driver = {}.{}\n",
        driver_ver / 1000,
        (driver_ver % 1000) / 10
    );

    println!("Result = PASS");
}
