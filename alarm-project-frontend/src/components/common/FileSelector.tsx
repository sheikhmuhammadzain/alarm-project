import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { getAllFiles, getAutoFiles } from '@/api/client';
import { useAppStore } from '@/store/app-store';
import type { FileInfo, FolderGroup } from '@/api/types';
import { cn } from '@/lib/utils';

interface FileSelectorProps {
  onFileSelect: (file: FileInfo | null) => void;
  className?: string;
  allowEmpty?: boolean;
}

export function FileSelector({ onFileSelect, className, allowEmpty = false }: FileSelectorProps) {
  const { selectedFile, setSelectedFile } = useAppStore();
  const [selectedFolder, setSelectedFolder] = useState<string>('');
  const [availableFiles, setAvailableFiles] = useState<FileInfo[]>([]);

  // Fetch all files
  const { data: allFiles, isLoading: filesLoading, error: filesError } = useQuery({
    queryKey: ['all-files'],
    queryFn: getAllFiles,
  });

  // Fetch auto files for folder organization
  const { data: autoFiles, isLoading: autoLoading } = useQuery({
    queryKey: ['auto-files'],
    queryFn: getAutoFiles,
  });

  // Create folder options
  const folderOptions = autoFiles?.map(folder => ({
    value: folder.folder_name,
    label: `${folder.folder_name} (${folder.file_count} files)`,
  })) || [];

  // Add "All Files" and "Uploaded Files" options
  const allOptions = [
    { value: 'all', label: 'All Files' },
    { value: 'uploaded', label: 'Uploaded Files Only' },
    ...folderOptions,
  ];

  // Update available files based on folder selection
  useEffect(() => {
    if (!allFiles) return;

    let filtered: FileInfo[] = [];
    
    if (selectedFolder === 'all' || selectedFolder === '') {
      filtered = allFiles;
    } else if (selectedFolder === 'uploaded') {
      filtered = allFiles.filter(file => file.source_type === 'uploaded');
    } else {
      // Filter by specific folder
      filtered = allFiles.filter(file => file.folder === selectedFolder);
    }
    
    setAvailableFiles(filtered);
  }, [allFiles, selectedFolder]);

  // Handle folder change
  const handleFolderChange = (folder: string) => {
    setSelectedFolder(folder);
    // Reset file selection when folder changes
    if (selectedFile && !availableFiles.find(f => f.file_id === selectedFile.file_id)) {
      handleFileChange('');
    }
  };

  // Handle file selection
  const handleFileChange = (fileId: string) => {
    if (!fileId) {
      onFileSelect(null);
      setSelectedFile(null);
      return;
    }
    
    const file = availableFiles.find(f => f.file_id === fileId);
    if (file) {
      onFileSelect(file);
      setSelectedFile(file);
    }
  };

  if (filesLoading || autoLoading) {
    return (
      <div className={cn('space-y-4', className)}>
        <div className="space-y-2">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-10 w-full" />
        </div>
        <div className="space-y-2">
          <Skeleton className="h-4 w-20" />
          <Skeleton className="h-10 w-full" />
        </div>
      </div>
    );
  }

  if (filesError) {
    return (
      <Alert variant="destructive" className={className}>
        <AlertDescription>
          Failed to load files. Please check your connection and try again.
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <div className={cn('space-y-4', className)}>
      {/* Folder/Source Selection */}
      <div className="space-y-2">
        <Label htmlFor="folder-select">Data Source</Label>
        <Select value={selectedFolder} onValueChange={handleFolderChange}>
          <SelectTrigger>
            <SelectValue placeholder="Select data source..." />
          </SelectTrigger>
          <SelectContent>
            {allOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                {option.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* File Selection */}
      <div className="space-y-2">
        <Label htmlFor="file-select">
          File ({availableFiles.length} available)
        </Label>
        <Select 
          value={selectedFile?.file_id || ''} 
          onValueChange={handleFileChange}
          disabled={availableFiles.length === 0}
        >
          <SelectTrigger>
            <SelectValue placeholder={
              availableFiles.length === 0 
                ? 'No files available'
                : 'Select a file to analyze...'
            } />
          </SelectTrigger>
          <SelectContent>
            {allowEmpty && (
              <SelectItem value="">
                <span className="text-muted-foreground">None selected</span>
              </SelectItem>
            )}
            {availableFiles.map((file) => (
              <SelectItem key={file.file_id} value={file.file_id}>
                <div className="flex items-center justify-between w-full">
                  <span className="truncate">{file.filename}</span>
                  <div className="flex items-center gap-2 ml-2">
                    <span className={cn(
                      'text-xs px-2 py-1 rounded-full',
                      file.source_type === 'uploaded' 
                        ? 'bg-isa-primary/10 text-isa-primary'
                        : 'bg-isa-success/10 text-isa-success'
                    )}>
                      {file.source_type === 'uploaded' ? 'Uploaded' : file.folder}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {file.size_mb.toFixed(1)} MB
                    </span>
                  </div>
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      
      {selectedFile && (
        <div className="rounded-lg border border-border bg-muted/50 p-3">
          <div className="text-sm">
            <p className="font-medium">{selectedFile.filename}</p>
            <div className="mt-1 flex items-center gap-4 text-xs text-muted-foreground">
              <span>Size: {selectedFile.size_mb.toFixed(1)} MB</span>
              <span>Source: {selectedFile.source_type}</span>
              {selectedFile.folder && <span>Folder: {selectedFile.folder}</span>}
              {selectedFile.row_count && <span>Rows: {selectedFile.row_count.toLocaleString()}</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}