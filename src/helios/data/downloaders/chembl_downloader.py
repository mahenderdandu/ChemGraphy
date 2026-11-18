"""
ChEMBL data downloader
Downloads and extracts data from ChEMBL database
"""

import os
import logging
from typing import Optional
import requests
from pathlib import Path
import gzip
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ChEMBLDownloader:
    """Download ChEMBL database files"""
    
    def __init__(self, output_dir: str, version: str = "33"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.base_url = f"https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_{version}/"
    
    def download_file(self, filename: str, force: bool = False) -> Path:
        """
        Download a file from ChEMBL FTP
        
        Args:
            filename: Name of file to download
            force: Force re-download if file exists
        
        Returns:
            Path to downloaded file
        """
        output_path = self.output_dir / filename
        
        if output_path.exists() and not force:
            logger.info(f"File {filename} already exists, skipping download")
            return output_path
        
        url = self.base_url + filename
        logger.info(f"Downloading {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded {filename} successfully")
            return output_path
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            if output_path.exists():
                output_path.unlink()
            raise
    
    def download_sqlite(self, force: bool = False) -> Path:
        """Download ChEMBL SQLite database"""
        filename = f"chembl_{self.version}_sqlite.tar.gz"
        archive_path = self.download_file(filename, force)
        
        # Extract if needed
        db_name = f"chembl_{self.version}.db"
        db_path = self.output_dir / db_name
        
        if not db_path.exists():
            logger.info(f"Extracting {archive_path}")
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(self.output_dir)
            logger.info(f"Extracted to {db_path}")
        
        return db_path
    
    def download_sdf(self, force: bool = False) -> Path:
        """Download ChEMBL SDF file"""
        filename = f"chembl_{self.version}.sdf.gz"
        return self.download_file(filename, force)
    
    def download_activity_data(self, force: bool = False) -> Path:
        """Download ChEMBL activity data (bioactivity)"""
        # Note: ChEMBL provides this through the SQLite database
        # This method is for convenience
        return self.download_sqlite(force)


def download_chembl_data(
    output_dir: str,
    version: str = "33",
    download_sqlite: bool = True,
    download_sdf: bool = False,
    force: bool = False
) -> dict:
    """
    Convenience function to download ChEMBL data
    
    Args:
        output_dir: Output directory
        version: ChEMBL version
        download_sqlite: Download SQLite database
        download_sdf: Download SDF file
        force: Force re-download
    
    Returns:
        Dictionary of downloaded file paths
    """
    downloader = ChEMBLDownloader(output_dir, version)
    
    downloads = {}
    
    if download_sqlite:
        downloads['sqlite'] = downloader.download_sqlite(force)
    
    if download_sdf:
        downloads['sdf'] = downloader.download_sdf(force)
    
    return downloads


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Download ChEMBL data')
    parser.add_argument('--output-dir', type=str, default='data/raw/chembl',
                       help='Output directory')
    parser.add_argument('--version', type=str, default='33',
                       help='ChEMBL version')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    files = download_chembl_data(
        args.output_dir,
        args.version,
        download_sqlite=True,
        download_sdf=False,
        force=args.force
    )
    
    print("\nDownloaded files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
