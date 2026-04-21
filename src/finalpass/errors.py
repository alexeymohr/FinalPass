from __future__ import annotations


class FinalPassError(Exception):
    """Base class for all FinalPass-raised errors caught by the CLI."""


class SpecError(FinalPassError):
    """Spec file is missing, malformed, or fails validation."""


class AudioFormatError(FinalPassError):
    """Input file cannot be read as a supported audio format."""


class ChannelMismatchError(FinalPassError):
    """File's channel count does not match the spec's channel_config."""


class LoudnessError(FinalPassError):
    """Loudness measurement could not be completed."""


class ClassifierConfigError(FinalPassError):
    """Classifier patterns YAML is missing, malformed, or fails validation."""


class AmbiguousClassificationError(FinalPassError):
    """A filename matched two or more role patterns."""


class DuplicateRoleError(FinalPassError):
    """A group has more than one file with the same role."""


class NoAudioFilesError(FinalPassError):
    """The target folder contains no readable WAV/BWF files."""


class SampleRateMismatchError(FinalPassError):
    """A set of files does not share one sample rate."""


class SampleCountMismatchError(FinalPassError):
    """A set of files does not share one total sample count."""


class AlignmentError(FinalPassError):
    """A constant non-zero global offset was detected between inputs."""


class UnsupportedChannelConfigError(FinalPassError):
    """A file's channel count is outside FinalPass v0.1 support."""


class ChannelConfigLabelMismatch(FinalPassError):
    """Filename channel label and file-header channel layout disagree."""


class ReportRenderError(FinalPassError):
    """HTML report rendering failed."""


class AAFExportError(FinalPassError):
    """AAF marker export failed."""
