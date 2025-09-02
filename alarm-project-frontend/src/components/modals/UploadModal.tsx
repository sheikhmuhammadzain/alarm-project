import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Upload, File, CheckCircle, AlertTriangle } from 'lucide-react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { uploadFile } from '@/api/client';
import { toast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';

interface UploadModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function UploadModal({ open, onOpenChange }: UploadModalProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const queryClient = useQueryClient();

  const uploadMutation = useMutation({
    mutationFn: uploadFile,
    onSuccess: (data) => {
      toast({
        title: 'Upload Successful',
        description: `${selectedFile?.name} has been uploaded successfully.`,
      });
      
      // Invalidate and refetch files data
      queryClient.invalidateQueries({ queryKey: ['files'] });
      queryClient.invalidateQueries({ queryKey: ['all-files'] });
      
      // Close modal and reset state
      onOpenChange(false);
      setSelectedFile(null);
      setUploadProgress(0);
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || 'Upload failed';
      toast({
        title: 'Upload Failed',
        description: message,
        variant: 'destructive',
      });
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Validate file type
      const allowedTypes = [
        'text/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      ];
      
      if (!allowedTypes.includes(file.type) && !file.name.toLowerCase().endsWith('.csv')) {
        toast({
          title: 'Invalid File Type',
          description: 'Please select a CSV or Excel file.',
          variant: 'destructive',
        });
        return;
      }
      
      // Validate file size (50MB limit)
      if (file.size > 50 * 1024 * 1024) {
        toast({
          title: 'File Too Large',
          description: 'File size must be less than 50MB.',
          variant: 'destructive',
        });
        return;
      }
      
      setSelectedFile(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: false,
  });

  const handleUpload = () => {
    if (!selectedFile) return;
    
    // Simulate progress for better UX
    const interval = setInterval(() => {
      setUploadProgress((prev) => {
        if (prev >= 90) {
          clearInterval(interval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);
    
    uploadMutation.mutate(selectedFile, {
      onSettled: () => {
        clearInterval(interval);
        setUploadProgress(100);
      },
    });
  };

  const handleClose = () => {
    if (uploadMutation.isPending) return;
    onOpenChange(false);
    setSelectedFile(null);
    setUploadProgress(0);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Upload Alarm Data File</DialogTitle>
          <DialogDescription>
            Upload a CSV or Excel file containing alarm data for analysis.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* File Drop Zone */}
          <div
            {...getRootProps()}
            className={cn(
              'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors',
              isDragActive
                ? 'border-isa-primary bg-isa-primary/5'
                : 'border-border hover:border-isa-primary hover:bg-muted/50',
              uploadMutation.isPending && 'pointer-events-none opacity-50'
            )}
          >
            <input {...getInputProps()} />
            
            {selectedFile ? (
              <div className="space-y-2">
                <CheckCircle className="mx-auto h-12 w-12 text-status-good" />
                <div className="space-y-1">
                  <p className="font-medium">{selectedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <Upload className="mx-auto h-12 w-12 text-muted-foreground" />
                <div>
                  <p className="font-medium">
                    {isDragActive ? 'Drop the file here' : 'Click to upload or drag and drop'}
                  </p>
                  <p className="text-sm text-muted-foreground">
                    CSV, XLS, or XLSX files (max 50MB)
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Upload Progress */}
          {uploadMutation.isPending && (
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Uploading...</span>
                <span>{uploadProgress}%</span>
              </div>
              <Progress value={uploadProgress} className="w-full" />
            </div>
          )}

          {/* Success/Error Messages */}
          {uploadMutation.isSuccess && (
            <Alert>
              <CheckCircle className="h-4 w-4" />
              <AlertDescription>
                File uploaded successfully! You can now analyze the data.
              </AlertDescription>
            </Alert>
          )}

          {uploadMutation.isError && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertDescription>
                {uploadMutation.error?.message || 'Upload failed. Please try again.'}
              </AlertDescription>
            </Alert>
          )}

          {/* File Requirements */}
          <div className="rounded-lg bg-muted/50 p-4">
            <h4 className="text-sm font-semibold mb-2">File Requirements</h4>
            <ul className="text-sm text-muted-foreground space-y-1">
              <li>• Format: CSV, XLS, or XLSX</li>
              <li>• Size: Maximum 50MB</li>
              <li>• Content: Should contain alarm timestamp, tag, and status columns</li>
              <li>• Encoding: UTF-8 recommended for CSV files</li>
            </ul>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={uploadMutation.isPending}>
            Cancel
          </Button>
          <Button
            onClick={handleUpload}
            disabled={!selectedFile || uploadMutation.isPending}
          >
            {uploadMutation.isPending ? 'Uploading...' : 'Upload File'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}