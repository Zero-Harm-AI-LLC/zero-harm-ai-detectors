#!/usr/bin/env python3
"""
Script to bump version and create a release
Usage: python scripts/bump_version.py [major|minor|patch]
"""

import sys
import re
import subprocess
from pathlib import Path

def get_current_version():
    """Get current version from __init__.py"""
    init_file = Path("zero_harm_detectors/__init__.py")
    content = init_file.read_text()
    match = re.search(r'__version__ = "([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Could not find version in __init__.py")

def bump_version(current_version, bump_type):
    """Bump version based on type"""
    major, minor, patch = map(int, current_version.split('.'))
    
    if bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'patch':
        patch += 1
    else:
        raise ValueError("bump_type must be 'major', 'minor', or 'patch'")
    
    return f"{major}.{minor}.{patch}"

def update_version_in_file(new_version):
    """Update version in __init__.py"""
    init_file = Path("zero_harm_detectors/__init__.py")
    content = init_file.read_text()
    new_content = re.sub(
        r'__version__ = "[^"]+"',
        f'__version__ = "{new_version}"',
        content
    )
    init_file.write_text(new_content)

def update_changelog(new_version):
    """Update CHANGELOG.md with new version"""
    changelog = Path("CHANGELOG.md")
    content = changelog.read_text()
    
    # Replace [Unreleased] with the new version
    today = subprocess.check_output(['date', '+%Y-%m-%d']).decode().strip()
    new_content = content.replace(
        "## [Unreleased]",
        f"## [Unreleased]\n\n## [{new_version}] - {today}"
    )
    
    changelog.write_text(new_content)

def create_git_tag(version):
    """Create and push git tag"""
    tag = f"v{version}"
    
    # Add changes
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', f'Bump version to {version}'], check=True)
    
    # Create tag
    subprocess.run(['git', 'tag', tag], check=True)
    
    # Push changes and tag
    subprocess.run(['git', 'push'], check=True)
    subprocess.run(['git', 'push', 'origin', tag], check=True)
    
    print(f"✅ Created and pushed tag {tag}")
    print(f"🚀 This will trigger automatic PyPI release and backend deployment")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ['major', 'minor', 'patch']:
        print("Usage: python scripts/bump_version.py [major|minor|patch]")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    
    try:
        current_version = get_current_version()
        new_version = bump_version(current_version, bump_type)
        
        print(f"Current version: {current_version}")
        print(f"New version: {new_version}")
        
        # Confirm
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("Cancelled")
            return
        
        # Update files
        update_version_in_file(new_version)
        update_changelog(new_version)
        
        # Create git tag and push
        create_git_tag(new_version)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()