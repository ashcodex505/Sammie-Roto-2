import os, sys, subprocess, platform, tomllib, shutil, shlex, tarfile, tempfile, plistlib
import urllib.request

# ===== CONFIG =====
PYTHON_VERSION = "3.12"
REPO_URL = "https://github.com/Zarxrax/Sammie-Roto-2.git"
REPO_RAW_BASE = "https://raw.githubusercontent.com/Zarxrax/Sammie-Roto-2"
DEFAULT_BRANCH = "main"
MLX_SAM_COMMIT = "fbeafa7531be454b8f42d89c13d1e90024a63f6b"
MLX_SAM_ARCHIVE = f"https://github.com/avbiswas/sam2-mlx/archive/{MLX_SAM_COMMIT}.tar.gz"


def ensure_mlx_sam_runtime():
    """Materialize the pinned optional MLX runtime on Apple Silicon.

    Git submodules cover normal clones. The exact-commit archive fallback also
    supports GitHub source archives and Sammie's installer/update workflow.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return
    app_dir = os.path.abspath(os.path.dirname(__file__))
    destination = os.path.join(app_dir, "third_party", "sam2-mlx")
    marker = os.path.join(destination, "src", "mlx_sam", "video_predictor.py")
    if os.path.isfile(marker):
        return

    # In a real clone, let Git materialize the gitlink recorded by this
    # repository. This keeps `git pull --rebase` and later submodule updates
    # clean. GitHub source archives do not include `.git`, so they use the
    # exact-commit archive fallback below.
    if os.path.isdir(os.path.join(app_dir, ".git")):
        try:
            subprocess.run(
                [
                    "git", "-C", app_dir, "submodule", "update", "--init",
                    "--recursive", "--", "third_party/sam2-mlx",
                ],
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"[Submodule initialization failed; using archive fallback: {error}]")
        if os.path.isfile(marker):
            return

    print("[Installing pinned SAM 2 MLX runtime...]")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sammie-mlx-") as temp_dir:
        archive = os.path.join(temp_dir, "runtime.tar.gz")
        urllib.request.urlretrieve(MLX_SAM_ARCHIVE, archive)
        unpacked = os.path.join(temp_dir, "unpacked")
        os.makedirs(unpacked)
        with tarfile.open(archive, "r:gz") as bundle:
            root = os.path.realpath(unpacked)
            for member in bundle.getmembers():
                target = os.path.realpath(os.path.join(unpacked, member.name))
                if os.path.commonpath((root, target)) != root:
                    raise RuntimeError("Unsafe path in SAM 2 MLX runtime archive")
            bundle.extractall(unpacked, filter="data")
        candidates = [
            path for path in os.listdir(unpacked)
            if os.path.isdir(os.path.join(unpacked, path))
        ]
        if len(candidates) != 1:
            raise RuntimeError("Unexpected SAM 2 MLX archive layout")
        source = os.path.join(unpacked, candidates[0])
        if os.path.isdir(destination):
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    if not os.path.isfile(marker):
        raise RuntimeError("SAM 2 MLX runtime installation did not complete")

def get_uv_exe():
    """Absolute path to the uv executable this installer bootstrapped with."""
    app_dir = os.path.abspath(os.path.dirname(__file__))
    exe_name = "uv.exe" if platform.system() == "Windows" else "uv"
    return os.path.join(app_dir, ".uv", exe_name)

def get_uv_env():
    """Environment for every uv subprocess manage.py invokes itself.
    Clears VIRTUAL_ENV so uv's internal build-isolation environment
    doesn't leak into child processes."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    return env

def fix_permissions():
    """Make launch scripts executable and clear download quarantine (macOS)."""
    if platform.system() == "Windows":
        return
    app_dir = os.path.abspath(os.path.dirname(__file__))

    for name in ("run_sammie.sh", "install.sh"):
        path = os.path.join(app_dir, name)
        if os.path.exists(path):
            try:
                os.chmod(path, os.stat(path).st_mode | 0o755)
            except OSError as e:
                print(f"[Warning: could not chmod {name}: {e}]")

    if platform.system() == "Darwin":
        # Skip the big generated folders; only the extracted download can be quarantined.
        skip = {".venv", ".uv", ".git"}
        for entry in os.listdir(app_dir):
            if entry in skip:
                continue
            subprocess.run(
                ["xattr", "-dr", "com.apple.quarantine", os.path.join(app_dir, entry)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )

def cleanup_cache():
    """After a successful install/reinstall/update, prunes the project cache."""
    run_command([get_uv_exe(), "cache", "prune"])

# ===== UTILS =====
def run_command(cmd):
    """Wrapper to handle uv commands."""
    print(">", " ".join(cmd))
    try:
        subprocess.check_call(cmd, env=get_uv_env())
    except subprocess.CalledProcessError as e:
        print(f"\nError executing command: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: 'uv' command not found. Please execute install.bat/install.sh")
        sys.exit(1)

def get_local_version():
    with open("pyproject.toml", "rb") as f:
        data = tomllib.load(f)
        return data["project"]["version"]

def get_remote_version(branch):
    raw_pyproject_url = f"{REPO_RAW_BASE}/{branch}/pyproject.toml"
    try:
        with urllib.request.urlopen(raw_pyproject_url) as response:
            data = tomllib.loads(response.read().decode())
            return data["project"]["version"]
    except Exception as e:
        print(f"[Warning: Could not check remote version: {e}]")
        return None

def parse_version(v):
    """Splits a dotted version string into a tuple of ints,
    e.g. '1.10.2' -> (1, 10, 2)."""
    parts = []
    for segment in v.strip().split("."):
        digits = ""
        for ch in segment:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits) if digits else 0)
    return tuple(parts)

def is_newer_version(remote_v, local_v):
    """Returns True if remote_v is a newer release than local_v."""
    r, l = parse_version(remote_v), parse_version(local_v)
    length = max(len(r), len(l))
    r += (0,) * (length - len(r))
    l += (0,) * (length - len(l))
    return r > l

def get_installed_backend():
    """Detects which torch extra is currently installed (used for updates)."""
    if platform.system() == "Darwin":
        return "macos"

    if not os.path.exists(".venv"):
        return None

    try:
        result = subprocess.check_output(
            [get_uv_exe(), "pip", "show", "torch", "--python", ".venv"], 
            text=True, stderr=subprocess.DEVNULL, env=get_uv_env()
        )
        version_line = next((l for l in result.splitlines() if l.startswith("Version:")), "").lower()
        for backend in ["cu130", "cu126", "rocm", "xpu", "cpu"]:
            if backend in version_line:
                return backend
    except (subprocess.CalledProcessError, StopIteration):
        print("[Warning: Could not detect installed backend]")
        pass
    
    return None

# ===== GIT LOGIC =====
def init_git_tracking():
    """Initializes git tracking for the install (adds an 'origin' remote
    pointing at REPO_URL) without fetching or touching any local files.
    Safe to call any time — does nothing if .git already exists."""
    from dulwich.repo import Repo
    from dulwich import porcelain

    if os.path.exists(".git"):
        return

    print("[Initializing Git tracking...]")
    repo = Repo.init(".")
    porcelain.remote_add(repo, "origin", REPO_URL)

def pull_latest_code(branch):
    """Ensures the local files exactly match the given branch of the repository"""
    from dulwich import porcelain
    from dulwich.repo import Repo

    init_git_tracking()
    repo = Repo(".")

    print(f"[Fetching latest code from GitHub ({branch} branch)...]")
    porcelain.fetch(repo, "origin")

    print("[Restoring all program files to original state...]")
    porcelain.reset(repo, "hard", f"origin/{branch}")

    # The hard reset can rewrite files with their git-stored mode, so
    # re-apply the exec bit and clear quarantine.
    fix_permissions()

# ===== BACKEND SELECTION =====
BACKEND_OPTIONS = [
    ("cu130", "NVIDIA CUDA 13.0 (RTX, newer GPUs)"),
    ("cu126", "NVIDIA CUDA 12.6 (GTX, older GPUs)"),
    ("xpu",   "Intel Arc/Xe (XPU)"),
    ("rocm",  "AMD ROCm"),
    ("cpu",   "CPU (Slow)"),
]

def choose_backend():
    """Manually prompt the user for their hardware backend."""
    if platform.system() == "Darwin":
        return "macos"

    print("\nSelect PyTorch backend:")
    for i, (_, label) in enumerate(BACKEND_OPTIONS, 1):
        print(f"{i}) {label}")

    choice = input( "> ").strip()
    try:
        return BACKEND_OPTIONS[int(choice) - 1][0]
    except (ValueError, IndexError):
        return "cpu"

def sync_env(backend, reinstall=False):
    """Uses uv sync to update or reinstall the environment."""
    cmd = [get_uv_exe(), "sync", "--frozen", "--python", PYTHON_VERSION, "--python-preference", "only-managed"]
    if backend:
        cmd.extend(["--extra", backend])
    
    if reinstall:
        print(f"\n[Reinstalling dependencies for {backend or 'Default/MPS'}...]")
        # --reinstall refreshes all;
        cmd.extend(["--reinstall"])
    else:
        print(f"\n[Syncing dependencies for {backend or 'Default/MPS'}...]")

    run_command(cmd)

def create_shortcuts():
    """Creates/refreshes the appropriate desktop entry point for the current OS."""
    system = platform.system()
    if system == "Windows":
        create_windows_shortcut()
    elif system == "Darwin":
        create_mac_app()
    elif system == "Linux":
        create_linux_desktop_entry()

def resolve_backend(context="continue"):
    """Returns the currently installed backend, or prompts the user to pick
    one if it can't be detected (e.g. after a failed/partial install)."""
    backend = get_installed_backend()
    if not backend:
        if platform.system() != "Darwin":
            print(f"[Could not determine your previously installed backend — please reselect it to {context}.]")
        backend = choose_backend()
    return backend

def perform_update(branch):
    """Pulls the given branch, resyncs the environment, and refreshes
    shortcuts. Shared by both update paths in handle_update()."""
    pull_latest_code(branch)
    ensure_mlx_sam_runtime()
    sync_env(resolve_backend())
    create_shortcuts()
    cleanup_cache()
    print("\nUpdate complete!")

# ===== CORE ACTIONS =====
def handle_update(branch):
    # Read the local version, with recovery if pyproject.toml is missing
    # or unreadable — a likely sign of a failed or partial install.
    try:
        local_v = get_local_version()
    except Exception as e:
        print(f"[Could not read local version: {e}]")
        print("[pyproject.toml may be missing or corrupt — this can happen after a failed install.]")
        recover = input("Pull latest code from GitHub to recover? (Y/n): ").strip().lower()
        if recover != "n":
            pull_latest_code(branch)
            sync_env(resolve_backend("continue recovery"))
            cleanup_cache()
            print("[Recovery complete!]")
        else:
            print("[No changes made. Consider using Reinstall/Repair from the main menu.]")
        return

    # Version numbers aren't a meaningful signal on a non-release branch --
    # always pull whatever's currently there instead of gating on them.
    if branch != DEFAULT_BRANCH:
        print(f"[Pulling latest '{branch}' branch...]")
        perform_update(branch)
        return

    remote_v = get_remote_version(branch)

    if remote_v is None:
        print("[Could not check for updates. Check your internet connection and try again.]")
        return

    if is_newer_version(remote_v, local_v):
        print(f"\nUpdate available: {remote_v} (current: {local_v})")
        if input("Install update now? (Y/n): ").strip().lower() == "n":
            print("Update skipped.")
            return
        perform_update(branch)
    else:
        print(f"[Already up to date (Version {local_v}).]")

def setup(branch, reinstall=False):
    # Git tracking is initialized as soon as setup() runs (fresh install or
    # reinstall), so future "Check for Updates" runs can fetch/reset
    # cleanly. This only adds the 'origin' remote -- it never touches files.
    init_git_tracking()

    # -- Gather all choices upfront ----------------------------------------

    # 1. Backend selection (with re-prompt on invalid input)
    backend = choose_backend()

    # 2. Pull latest code?
    if reinstall:
        prompt = (
            "\nAlso pull the latest code from GitHub? This will overwrite "
            "any local changes to program files. (y/N): "
        )
        pull_code = input(prompt).strip().lower() == "y"
    else:
        prompt = (
            "\nAlso pull the latest code from GitHub? Only needed if you think "
            "the downloaded files are out of date. (y/N): "
        )
        pull_code = input(prompt).strip().lower() == "y"

    # 3. Model download -- fresh install only
    download_models_now = False
    if not reinstall:
        print("\nModel download:")
        print("1) Download models as needed (default -- models download the first time they are used)")
        print("2) Download all models now (~10GB)")
        model_choice = input("> ").strip()
        download_models_now = model_choice == "2"

    # -- Summarise and confirm ----------------------------------------------
    backend_labels = dict(BACKEND_OPTIONS)

    print("\n--- Setup summary ---")
    print(f"  Branch           : {branch}")
    print(f"  Pull latest code : {'Yes' if pull_code else 'No'}")
    if platform.system() != "Darwin":
        print(f"  PyTorch backend  : {backend_labels.get(backend, backend)}")
    if not reinstall:
        print(f"  Download models  : {'Download all now (~10GB)' if download_models_now else 'Download as needed'}")
    print("---------------------")

    confirm = input("\nProceed with setup? (Y/n): ").strip().lower()
    if confirm == "n":
        print("Setup cancelled.")
        sys.exit(0)

    # -- Execute ------------------------------------------------------------
    if pull_code:
        pull_latest_code(branch)

    # Always make scripts executable / clear quarantine, even if the
    # dependency install below fails partway.
    fix_permissions()

    ensure_mlx_sam_runtime()

    run_command([get_uv_exe(), "python", "install", "--no-bin", PYTHON_VERSION])

    sync_env(backend, reinstall=reinstall)

    create_shortcuts()

    cleanup_cache()

    # Run the model downloader last so all dependencies are in place.
    if download_models_now:
        print("\nDownloading all models...")
        run_command([get_uv_exe(), "run", os.path.join("sammie", "model_downloader.py")])

    print("\nSetup Complete!")


# ===== CREATE SHORTCUTS =====
def create_mac_app():
    """Create a Spotlight-visible launcher in the user's Applications folder."""

    app_dir = os.path.abspath(os.path.dirname(__file__))
    user_apps = os.path.join(os.path.expanduser("~"), "Applications")
    os.makedirs(user_apps, exist_ok=True)
    app_bundle = os.path.join(user_apps, "Sammie-Roto-2.app")

    # Older installers put the bundle beside the source tree and symlinked it
    # into ~/Applications. Spotlight does not consistently index application
    # symlinks, so replace that link with a real bundle in the canonical app
    # location. The small bundle still launches the selected Sammie checkout.
    if os.path.islink(app_bundle):
        os.unlink(app_bundle)
    elif os.path.isdir(app_bundle):
        shutil.rmtree(app_bundle)

    log_path = os.path.join(os.path.expanduser("~"), "Library", "Logs", "Sammie-Roto-2.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    shell_command = (
        f"cd {shlex.quote(app_dir)} && "
        "exec /bin/bash ./run_sammie.sh "
        f">> {shlex.quote(log_path)} 2>&1 </dev/null &"
    )
    applescript_command = shell_command.replace("\\", "\\\\").replace('"', '\\"')
    subprocess.run(
        [
            "osacompile", "-o", app_bundle, "-e",
            f'do shell script "{applescript_command}"',
        ],
        check=True,
    )

    resources_dir = os.path.join(app_bundle, "Contents", "Resources")
    version = get_local_version()  # pulls from pyproject.toml

    src_icon = os.path.join(app_dir, "sammie", "resources", "icon.icns")
    dest_icon = os.path.join(resources_dir, "icon.icns")
    if os.path.exists(src_icon):
        shutil.copy(src_icon, dest_icon)

    # Preserve the runtime keys generated by osacompile and add Sammie's app
    # identity. The applet executable is a native universal Mach-O binary, so
    # Launch Services handles it more reliably than a shell script executable.
    plist_path = os.path.join(app_bundle, "Contents", "Info.plist")
    with open(plist_path, "rb") as f:
        app_info = plistlib.load(f)
    app_info.update({
        "CFBundleName": "Sammie-Roto-2",
        "CFBundleDisplayName": "Sammie Roto 2",
        "CFBundleIconFile": "icon.icns",
        "CFBundleIdentifier": "com.zarxrax.sammie-roto-2",
        "CFBundleVersion": version,
        "CFBundleShortVersionString": version,
        "LSApplicationCategoryType": "public.app-category.video",
        "NSHighResolutionCapable": True,
    })
    with open(plist_path, "wb") as f:
        plistlib.dump(app_info, f)

    # Clear quarantine/provenance metadata inherited from downloaded source
    # files. Finder and Spotlight may silently refuse an unsigned script-based
    # bundle when those attributes remain attached.
    try:
        subprocess.run(
            ["xattr", "-cr", app_bundle],
            check=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        pass

    # An ad-hoc signature is sufficient for a locally generated launcher. It
    # gives Launch Services a complete, internally consistent app bundle while
    # avoiding any requirement for an Apple Developer certificate.
    try:
        subprocess.run(
            ["codesign", "--force", "--deep", "--sign", "-", app_bundle],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[Warning: Could not sign macOS app bundle: {error}]")

    print(f"Created {app_bundle}")

    # Create a Desktop link to the canonical bundle for convenience.
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if os.path.isdir(desktop):
        desktop_link = os.path.join(desktop, "Sammie-Roto-2.app")
        try:
            if os.path.islink(desktop_link) or os.path.exists(desktop_link):
                os.remove(desktop_link)
            os.symlink(app_bundle, desktop_link)
            print(f"Created Desktop shortcut at: {desktop_link}")
        except OSError as e:
            print(f"[Warning: Could not create Desktop shortcut: {e}]")

    print(f"Added to your Applications folder: {app_bundle}")
    print("Double-click the Desktop icon or search Spotlight for 'Sammie Roto 2'.")


def create_linux_desktop_entry():
    """Creates a .desktop file for GNOME and KDE integration."""
    home = os.path.expanduser("~")
    apps_dir = os.path.join(home, ".local", "share", "applications")
    os.makedirs(apps_dir, exist_ok=True)
    
    desktop_path = os.path.join(apps_dir, "sammie-roto-2.desktop")
    app_dir = os.path.abspath(os.path.dirname(__file__))
    icon_path = os.path.join(app_dir, "sammie", "resources", "icon.png")
    run_sh_path = os.path.join(app_dir, "run_sammie.sh")

    content = [
        "[Desktop Entry]",
        "Type=Application",
        "Name=Sammie-Roto-2",
        "Comment=Video Rotoscoping and Masking Tool",
        f"Exec=\"{run_sh_path}\"",
        f"Icon={icon_path}",
        "Terminal=false",
        "Categories=Graphics;Video;VideoEditing;",
        "StartupWMClass=Sammie-Roto-2",
    ]

    with open(desktop_path, "w") as f:
        f.write("\n".join(content))
    
    os.chmod(desktop_path, 0o755)
    print(f"Created Linux desktop shortcut at: {desktop_path}")

def create_windows_shortcut():
    """Creates a desktop shortcut on Windows."""
    app_dir = os.path.abspath(os.path.dirname(__file__))
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    shortcut_path = os.path.join(desktop, "Sammie-Roto-2.lnk")
    target = os.path.join(app_dir, "run_sammie.bat")
    icon = os.path.join(app_dir, "sammie", "resources", "icon.ico")

    ps_script = (
        f'$ws = New-Object -ComObject WScript.Shell;'
        f'$s = $ws.CreateShortcut("{shortcut_path}");'
        f'$s.TargetPath = "{target}";'
        f'$s.WorkingDirectory = "{app_dir}";'
        f'$s.IconLocation = "{icon}";'
        f'$s.Save()'
    )

    try:
        subprocess.check_call(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print(f"Created Windows desktop shortcut at: {shortcut_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Warning: Could not create Windows shortcut: {e}]")

# ===== ENTRY =====
def is_app_running():
    """Checks whether Sammie-Roto is currently running by reading the PID
    from Qt's lock file and verifying the process is actually alive."""
    lock_path = os.path.join(
        os.environ.get("TEMP", os.environ.get("TMP", "")) if platform.system() == "Windows" else "/tmp",
        "sammie-roto.lock"
    )
    if not os.path.exists(lock_path):
        return False
    try:
        with open(lock_path, "r") as f:
            pid = int(f.readline().strip())
    except (ValueError, OSError):
        return False  # Unreadable or malformed — treat as stale

    if platform.system() == "Windows":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True
        )
        return str(pid) in result.stdout
    else:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False  # Process doesn't exist — stale lock
        except PermissionError:
            return True   # Process exists but we can't signal it — assume running

def main():
    dev_mode = "--dev" in sys.argv[1:]

    # If app is not installed, bypass main menu and proceed to setup
    if not os.path.exists(".venv"):
        if os.path.exists("python-3.12.8-embed-amd64"):
            print("ERROR: It appears you are trying to install over an older Sammie-Roto installation.")
            print("Please delete the existing folder then extract the files to a new folder and try again.")
            sys.exit(1)

        branch = DEFAULT_BRANCH
        if dev_mode:
            print("\nInstall from:")
            print("1) Main (stable)")
            print("2) Dev (latest, may be unstable)")
            branch = "dev" if input("> ").strip() == "2" else DEFAULT_BRANCH

        setup(branch)
        return

    if is_app_running():
        print("\n[Warning: Sammie-Roto-2 appears to be running.]")
        print("[Please close it before continuing to avoid corrupting your installation.]")
        confirm = input("Continue anyway? (y/N): ").strip().lower()
        if confirm != "y":
            sys.exit(0)

    print("\nSammie-Roto-2 Manager")

    actions = [("Check for Updates", lambda: handle_update(DEFAULT_BRANCH))]
    if dev_mode:
        actions.append(("Pull Latest Dev Commits", lambda: handle_update("dev")))
    actions.append(("Reinstall/Repair", lambda: setup(DEFAULT_BRANCH, reinstall=True)))
    actions.append(("Exit", lambda: sys.exit(0)))

    for i, (label, _) in enumerate(actions, 1):
        print(f"{i}) {label}")

    choice = input("> ").strip()
    try:
        index = int(choice) - 1
        if 0 <= index < len(actions):
            actions[index][1]()
        else:
            sys.exit(0)
    except ValueError:
        sys.exit(0)

if __name__ == "__main__":
    main()
