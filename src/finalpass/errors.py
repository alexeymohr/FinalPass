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
