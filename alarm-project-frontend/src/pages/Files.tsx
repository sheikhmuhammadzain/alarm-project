import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { DataTable, Column } from '@/components/common/DataTable';
import { UploadModal } from '@/components/modals/UploadModal';
import { PreviewModal } from '@/components/modals/PreviewModal';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { 
  Upload, 
  RefreshCw, 
  Eye, 
  Trash2, 
  Download,
  FolderOpen,
  Files as FilesIcon
} from 'lucide-react';
import { 
  getAllFiles, 
  getAutoFiles, 
  refreshAutoFiles, 
  deleteFile, 
  deleteAllFiles 
} from '@/api/client';
import { toast } from '@/hooks/use-toast';
import type { FileInfo } from '@/api/types';
import { cn } from '@/lib/utils';

export default function Files() {
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [previewModalOpen, setPreviewModalOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<FileInfo | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [deleteAllDialogOpen, setDeleteAllDialogOpen] = useState(false);
  const [fileToDelete, setFileToDelete] = useState<FileInfo | null>(null);
  const [sourceFilter, setSourceFilter] = useState<string>('all');

  const queryClient = useQueryClient();

  // Fetch all files
  const { data: allFiles = [], isLoading: filesLoading, error: filesError } = useQuery({
    queryKey: ['all-files'],
    queryFn: getAllFiles,
  });

  // Fetch auto files for folder info
  const { data: autoFiles = [], isLoading: autoLoading } = useQuery({
    queryKey: ['auto-files'],
    queryFn: getAutoFiles,
  });

  // Refresh auto files mutation
  const refreshMutation = useMutation({
    mutationFn: refreshAutoFiles,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['auto-files'] });
      queryClient.invalidateQueries({ queryKey: ['all-files'] });
      toast({
        title: 'Files Refreshed',
        description: 'Auto-discovered files have been refreshed successfully.',
      });
    },
    onError: (error: any) => {
      toast({
        title: 'Refresh Failed',
        description: error.response?.data?.detail || 'Failed to refresh files.',
        variant: 'destructive',
      });
    },
  });

  // Delete file mutation
  const deleteMutation = useMutation({
    mutationFn: deleteFile,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] });
      queryClient.invalidateQueries({ queryKey: ['all-files'] });
      toast({
        title: 'File Deleted',
        description: `${fileToDelete?.filename} has been deleted successfully.`,
      });
      setFileToDelete(null);
      setDeleteDialogOpen(false);
    },
    onError: (error: any) => {
      toast({
        title: 'Delete Failed',
        description: error.response?.data?.detail || 'Failed to delete file.',
        variant: 'destructive',
      });
    },
  });

  // Delete all files mutation
  const deleteAllMutation = useMutation({
    mutationFn: deleteAllFiles,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['files'] });
      queryClient.invalidateQueries({ queryKey: ['all-files'] });
      toast({
        title: 'All Files Deleted',
        description: 'All uploaded files have been deleted successfully.',
      });
      setDeleteAllDialogOpen(false);
    },
    onError: (error: any) => {
      toast({
        title: 'Delete Failed',
        description: error.response?.data?.detail || 'Failed to delete all files.',
        variant: 'destructive',
      });
    },
  });

  // Filter files based on source
  const filteredFiles = allFiles.filter(file => {
    if (sourceFilter === 'all') return true;
    if (sourceFilter === 'uploaded') return file.source_type === 'uploaded';
    return file.folder === sourceFilter;
  });

  // Get folder options for filter
  const folderOptions = [
    { value: 'all', label: 'All Sources' },
    { value: 'uploaded', label: 'Uploaded Files Only' },
    ...autoFiles.map(folder => ({
      value: folder.folder_name,
      label: `${folder.folder_name} (${folder.file_count})`
    }))
  ];

  // Table columns
  const columns: Column[] = [
    {
      key: 'filename',
      label: 'File Name',
      render: (value, row) => (
        <div className="flex items-center gap-2">
          <FilesIcon className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium">{value}</span>
        </div>
      ),
    },
    {
      key: 'source_type',
      label: 'Source',
      render: (value, row) => (
        <Badge variant={value === 'uploaded' ? 'default' : 'secondary'}>
          {value === 'uploaded' ? 'Uploaded' : row.folder || 'Auto-discovered'}
        </Badge>
      ),
    },
    {
      key: 'size_mb',
      label: 'Size',
      render: (value) => `${value.toFixed(1)} MB`,
    },
    {
      key: 'row_count',
      label: 'Rows',
      render: (value) => value ? value.toLocaleString() : '—',
    },
    {
      key: 'created_at',
      label: 'Created',
      render: (value) => new Date(value).toLocaleDateString(),
    },
    {
      key: 'actions',
      label: 'Actions',
      sortable: false,
      render: (_, row) => (
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setSelectedFile(row);
              setPreviewModalOpen(true);
            }}
          >
            <Eye className="h-4 w-4" />
          </Button>
          {row.source_type === 'uploaded' && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setFileToDelete(row);
                setDeleteDialogOpen(true);
              }}
            >
              <Trash2 className="h-4 w-4 text-destructive" />
            </Button>
          )}
        </div>
      ),
    },
  ];

  const uploadedFilesCount = allFiles.filter(f => f.source_type === 'uploaded').length;

  if (filesError) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Files</h1>
          <p className="text-muted-foreground">Manage your alarm data files</p>
        </div>
        
        <Alert variant="destructive">
          <AlertDescription>
            Failed to load files: {filesError.message}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-foreground">Files</h1>
          <p className="text-muted-foreground">
            Manage and organize your alarm data files
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => refreshMutation.mutate()}
            disabled={refreshMutation.isPending}
          >
            <RefreshCw className={cn(
              'h-4 w-4 mr-2',
              refreshMutation.isPending && 'animate-spin'
            )} />
            Refresh Auto Files
          </Button>
          
          <Button onClick={() => setUploadModalOpen(true)}>
            <Upload className="h-4 w-4 mr-2" />
            Upload File
          </Button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <FilesIcon className="h-5 w-5 text-isa-primary" />
            <h3 className="font-semibold">Total Files</h3>
          </div>
          <div className="mt-2 text-2xl font-bold">{allFiles.length}</div>
        </div>
        
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <Upload className="h-5 w-5 text-status-acceptable" />
            <h3 className="font-semibold">Uploaded Files</h3>
          </div>
          <div className="mt-2 text-2xl font-bold">{uploadedFilesCount}</div>
        </div>
        
        <div className="rounded-lg border border-border bg-card p-4">
          <div className="flex items-center gap-2">
            <FolderOpen className="h-5 w-5 text-status-good" />
            <h3 className="font-semibold">Auto-discovered</h3>
          </div>
          <div className="mt-2 text-2xl font-bold">{autoFiles.length} folders</div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">Filter by source:</span>
          <Select value={sourceFilter} onValueChange={setSourceFilter}>
            <SelectTrigger className="w-48">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {folderOptions.map(option => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {uploadedFilesCount > 0 && (
          <Button
            variant="outline"
            onClick={() => setDeleteAllDialogOpen(true)}
            disabled={deleteAllMutation.isPending}
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Delete All Uploaded
          </Button>
        )}
      </div>

      {/* Files Table */}
      {filesLoading ? (
        <div className="space-y-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-16 w-full" />
          ))}
        </div>
      ) : (
        <DataTable
          data={filteredFiles}
          columns={columns}
          searchable={true}
          sortable={true}
          paginated={true}
          pageSize={20}
          emptyMessage="No files found"
        />
      )}

      {/* Modals */}
      <UploadModal
        open={uploadModalOpen}
        onOpenChange={setUploadModalOpen}
      />
      
      <PreviewModal
        file={selectedFile}
        open={previewModalOpen}
        onOpenChange={setPreviewModalOpen}
      />

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialogOpen} onOpenChange={setDeleteDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete File</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{fileToDelete?.filename}"? 
              This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => fileToDelete && deleteMutation.mutate(fileToDelete.file_id)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete All Confirmation Dialog */}
      <AlertDialog open={deleteAllDialogOpen} onOpenChange={setDeleteAllDialogOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete All Uploaded Files</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete all {uploadedFilesCount} uploaded files? 
              This action cannot be undone and will not affect auto-discovered files.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteAllMutation.mutate()}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete All
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}