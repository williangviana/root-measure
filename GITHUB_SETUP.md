# Git & GitHub Setup Protocol

Quick reference for creating new projects and linking them to GitHub.

## Prerequisites

- Git installed on your computer
- GitHub account created
- SSH key configured with GitHub (or use HTTPS)

## Creating a New Project Repository

### 1. Create Project Folder
```bash
cd ~/path/to/Github/
mkdir project-name
cd project-name
```

### 2. Initialize Git Repository
```bash
git init
```

### 3. Create Your Files
Add your code, scripts, or documents to the folder.

### 4. Create First Commit
```bash
git add .
git commit -m "Initialize project"
```

## Linking to GitHub

### 5. Create GitHub Repository
Go to [github.com](https://github.com) and:
- Click the `+` icon → "New repository"
- Repository name: `project-name` (match your local folder)
- Choose Public or Private
- **DO NOT** initialize with README, .gitignore, or license
- Click "Create repository"

### 6. Link Local Repository to GitHub
```bash
git remote add origin git@github.com:YOUR-USERNAME/project-name.git
```

Replace `YOUR-USERNAME` with your GitHub username.

### 7. Push to GitHub
```bash
git push -u origin master
```

Or if using `main` as default branch:
```bash
git push -u origin main
```

## Common Git Commands

### Daily Workflow
```bash
# Check status
git status

# Stage files
git add filename.txt        # Add specific file
git add .                   # Add all changes

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push

# Pull latest changes
git pull
```

### Checking Configuration
```bash
# View remote URL
git remote -v

# View commit history
git log --oneline

# View current branch
git branch
```

## Example: Your RNA-seq Project

Your `rna-seq` project is set up as:
```
Location: ~/...Google Drive.../Code/Github/rna-seq/
GitHub: git@github.com:williangviana/rna-seq.git
```

New projects follow the same pattern:
```
~/...Google Drive.../Code/Github/root-measuring/ → github.com/williangviana/root-measuring
~/...Google Drive.../Code/Github/fluorescence/   → github.com/williangviana/fluorescence
```

## Troubleshooting

**"remote origin already exists"**
```bash
git remote remove origin
git remote add origin git@github.com:YOUR-USERNAME/project-name.git
```

**"Permission denied (publickey)"**
- Use HTTPS instead: `git remote add origin https://github.com/YOUR-USERNAME/project-name.git`
- Or set up SSH keys: [GitHub SSH Guide](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)

**"Branch 'master' vs 'main'"**
- GitHub uses `main` as default, older repos use `master`
- Rename if needed: `git branch -M main`

**Check if folder is already a git repo**
```bash
ls -la | grep .git
# If you see .git, it's already initialized
```

## Notes

- Each project folder is an independent repository
- The `Github/` parent folder is NOT a repository itself
- Each project needs its own GitHub repository
- Always commit before pushing to GitHub
